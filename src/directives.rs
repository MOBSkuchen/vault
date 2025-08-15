use crate::codeviz::print_code_warn;
use crate::comp_errors::{CodeError, CodeResult, CodeWarning};
use crate::DevDebugLevel;
use crate::filemanager::FileManager;
use crate::parser::{Directive, DirectiveArgType, DirectiveExpr, VirtualDirectiveArgType};
use crate::utils::dedup;

pub struct CompilationConfig {
    debug: bool,
    pub libs: Vec<String>,
    pub dev_debug_level: DevDebugLevel
}

impl CompilationConfig {
    pub fn new(debug: bool, dev_debug_level: DevDebugLevel) -> Self {
        Self {debug, libs: Vec::new(), dev_debug_level}
    }
}

fn directive_expr_to_virtual_type(directive_expr: &DirectiveExpr) -> VirtualDirectiveArgType {
    match directive_expr {
        DirectiveExpr::Literal(lit) => {
            match lit {
                DirectiveArgType::Identifier { .. } => VirtualDirectiveArgType::Identifier,
                DirectiveArgType::IntNumber { .. } => VirtualDirectiveArgType::IntNumber,
                DirectiveArgType::FloatNumber { .. } => VirtualDirectiveArgType::FloatNumber,
                DirectiveArgType::String { .. } => VirtualDirectiveArgType::String
            }
        }
        DirectiveExpr::NestedDirective(_) => VirtualDirectiveArgType::Directive,
    }
}

fn check_directive_signature(directive: &Directive, expected: Vec<VirtualDirectiveArgType>) -> CodeResult<()> {
    let virtuals = directive.arguments.iter().map(|x| {directive_expr_to_virtual_type(x)}).collect::<Vec<VirtualDirectiveArgType>>();
    if expected.iter().zip(virtuals.iter()).filter(|&(a, b)| a == b).count() != expected.len() {
        Err(CodeError::wrong_directive_arg_sig(directive.name, directive.code_position, expected, virtuals))
    } else {Ok(())}
}

macro_rules! count_idents {
    () => {0};
    ($_head:ident $($tail:ident)*) => {1 + count_idents!($($tail)*)};
}

macro_rules! virtual_directive_args {
    (
        directive = $directive:expr,
        args = [ $( $name:ident : $val:ident ),* $(,)? ],
        values = $values:expr,
    ) => {
            let expected = vec![
                $( VirtualDirectiveArgType::$val ),*
            ];
            check_directive_signature(&$directive, expected)?;

            debug_assert_eq!(count_idents!($($name)*) , $values.len(), "Args and values length mismatch");
            virtual_directive_args!(@index 0, $values, $( $name : $val ),*);
    };

    (@index $idx:expr, $values:expr, ) => {}; // end recursion

    (@index $idx:expr, $values:expr, $name:ident : $val:ident, $($rest_name:ident : $rest_val:ident),*) => {
        virtual_directive_args!(@handle $name, $val, &$values[$idx]);
        virtual_directive_args!(@index $idx + 1, $values, $($rest_name : $rest_val),*);
    };

    (@index $idx:expr, $values:expr, $name:ident : $val:ident) => {
        virtual_directive_args!(@handle $name, $val, &$values[$idx]);
    };

    (@handle $name:ident, IntNumber, $value:expr) => {
        let $name = {
            match $value {
                DirectiveExpr::Literal(lit) => match lit {
                    DirectiveArgType::IntNumber { value, .. } => value,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        };
    };

    (@handle $name:ident, Identifier, $value:expr) => {
        let $name = {
            match $value {
                DirectiveExpr::Literal(lit) => match lit {
                    DirectiveArgType::Identifier { value, token: _ } => value,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        };
    };

    (@handle $name:ident, FloatNumber, $value:expr) => {
        let $name = {
            match $value {
                DirectiveExpr::Literal(lit) => match lit {
                    DirectiveArgType::FloatNumber { value, .. } => value,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        };
    };

    (@handle $name:ident, String, $value:expr) => {
        let $name = {
            match $value {
                DirectiveExpr::Literal(lit) => match lit {
                    DirectiveArgType::String { value, token: _ } => value,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        };
    };

    (@handle $name:ident, Directive, $value:expr) => {
        let $name = {
            match $value {
                DirectiveExpr::NestedDirective(directive) => directive,
                _ => unreachable!(),
            }
        };
    };
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum FnSignals {
    Final,
    IgnoreReservedName
}

pub struct FunctionModifier {
    pub enable: bool,
    pub modifiers: Vec<FnSignals>
}

impl FunctionModifier {
    pub fn new(enable: bool) -> Self {
        Self { enable, modifiers: Vec::new()}
    }

    pub fn passthrough(m: FnSignals) -> Self {
        Self { enable: true, modifiers: vec![m]}
    }

    pub fn merge(&mut self, enable: bool, other: &mut FunctionModifier) {
        self.enable = enable;
        self.modifiers.append(&mut other.modifiers);
        dedup(&mut self.modifiers)
    }
}

pub fn visit_directive(directive: Directive, compilation_config: &mut CompilationConfig, file_manager: &FileManager) -> CodeResult<FunctionModifier> {
    let d = directive.name.content.to_uppercase();
    match d.as_str() {
        "AND" => {
            virtual_directive_args! {
                directive = directive,
                args = [ left: Directive, right: Directive ],
                values = directive.arguments,
            }
            let mut left = visit_directive(*left.clone(), compilation_config, file_manager)?;
            let mut right = visit_directive(*right.clone(), compilation_config, file_manager)?;
            left.merge(left.enable && right.enable, &mut right);
            Ok(left)
        }
        "OR" => {
            virtual_directive_args! {
                directive = directive,
                args = [ left: Directive, right: Directive ],
                values = directive.arguments,
            }
            let mut left = visit_directive(*left.clone(), compilation_config, file_manager)?;
            let mut right = visit_directive(*right.clone(), compilation_config, file_manager)?;
            left.merge(left.enable || right.enable, &mut right);
            Ok(left)
        }
        "XOR" => {
            virtual_directive_args! {
                directive = directive,
                args = [ left: Directive, right: Directive ],
                values = directive.arguments,
            }
            let mut left = visit_directive(*left.clone(), compilation_config, file_manager)?;
            let mut right = visit_directive(*right.clone(), compilation_config, file_manager)?;
            left.merge(left.enable ^ right.enable, &mut right);
            Ok(left)
        }
        "ALWAYS" => {
            virtual_directive_args! {
                directive = directive,
                args = [ ],
                values = directive.arguments,
            }
            Ok(FunctionModifier::new(true))
        }
        "NEVER" => {
            virtual_directive_args! {
                directive = directive,
                args = [ ],
                values = directive.arguments,
            }
            Ok(FunctionModifier::new(true))
        }
        "NOT" => {
            virtual_directive_args! {
                directive = directive,
                args = [ expr: Directive ],
                values = directive.arguments,
            }
            let mut left = visit_directive(*expr.clone(), compilation_config, file_manager)?;
            left.enable = !left.enable;
            Ok(left)
        }
        "OS" => {
            virtual_directive_args! {
                directive = directive,
                args = [ os: String ],
                values = directive.arguments,
            }
            Ok(FunctionModifier::new(std::env::consts::OS == os))
        }
        "ARCH" => {
            virtual_directive_args! {
                directive = directive,
                args = [ arch: String ],
                values = directive.arguments,
            }
            Ok(FunctionModifier::new(std::env::consts::ARCH == arch))
        }
        "DEBUG" => {
            virtual_directive_args! {
                directive = directive,
                args = [],
                values = directive.arguments,
            }
            Ok(FunctionModifier::new(compilation_config.debug))
        }
        "IGNORE_RESERVED" => {
            virtual_directive_args! {
                directive = directive,
                args = [],
                values = directive.arguments,
            }
            Ok(FunctionModifier::passthrough(FnSignals::IgnoreReservedName))
        }
        "FINAL" => {
            virtual_directive_args! {
                directive = directive,
                args = [],
                values = directive.arguments,
            }
            Ok(FunctionModifier::passthrough(FnSignals::Final))
        }
        "LINK" => {
            virtual_directive_args! {
                directive = directive,
                args = [ lib: String ],
                values = directive.arguments,
            }
            compilation_config.libs.push(lib.clone());
            Ok(FunctionModifier::new(true))
        }
        // IF && IF IMMUTABLE
        "IF" | "IFI" => {
            virtual_directive_args! {
                directive = directive,
                args = [ cond: Directive, exe: Directive,  ],
                values = directive.arguments,
            }
            let fmod = visit_directive(*cond.clone(), compilation_config, file_manager)?;
            let ret = if fmod.enable { visit_directive(*exe.clone(), compilation_config, file_manager)? } else { return Ok(fmod) };
            Ok(if d.ends_with("I") { fmod } else { ret })
        }
        // IF ELSE && IF ELSE IMMUTABLE
        "IFE" | "IFEI" => {
            virtual_directive_args! {
                directive = directive,
                args = [ cond: Directive, exe: Directive, otherwise: Directive ],
                values = directive.arguments,
            }
            let fmod = visit_directive(*cond.clone(), compilation_config, file_manager)?;
            let ret = if fmod.enable { visit_directive(*exe.clone(), compilation_config, file_manager)? } else { visit_directive(*otherwise.clone(), compilation_config, file_manager)? };
            Ok(if d.ends_with("I") { fmod } else { ret })
        }
        "ERR" => {
            virtual_directive_args! {
                directive = directive,
                args = [ msg: String ],
                values = directive.arguments,
            }
            Err(CodeError::error_from_directive(directive.code_position, msg.to_owned(), vec![]))
        }
        // Error note
        "ERRN" => {
            virtual_directive_args! {
                directive = directive,
                args = [ msg: String, note: String ],
                values = directive.arguments,
            }
            Err(CodeError::error_from_directive(directive.code_position, msg.to_owned(), vec![note.to_owned()]))
        }
        "WARN" | "WARNF" => {
            virtual_directive_args! {
                directive = directive,
                args = [ msg: String ],
                values = directive.arguments,
            }
            let warning = CodeWarning::directive_warning(directive.code_position, msg.to_owned());
            print_code_warn(warning, file_manager);
            Ok(FunctionModifier::new(!d.ends_with("F")))
        }
        _ => Err(CodeError::unknown_directive(directive.name)) 
    }
}