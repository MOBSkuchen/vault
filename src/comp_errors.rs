use crate::codeviz::print_code_error;
use crate::filemanager::FileManager;
use crate::lexer::{CodePosition, Token, TokenType};
use colorize_rs::AnsiColor;
use std::fmt;
use num_derive::FromPrimitive;
use crate::parser::{format_virtual_type_sig, Expression, ModuleAccessVariant, TypesKind, VirtualDirectiveArgType};

#[derive(Debug)]
pub enum CompilerError {
    FileNotAccessible(String, bool),
    FileCorrupted(String),
    VerifyFailed(String),
}

impl CompilerError {
    pub fn output(&self) {
        match self {
            CompilerError::FileNotAccessible(f, s) => {
                println!(
                    "Could not read input file `{}`, because the file is not accessible.",
                    f.to_string().bold().b_yellow().underlined()
                );
                if *s {
                    println!(
                        "{}",
                        "This is because the directory does not exist!".bold().red()
                    );
                }
            }
            CompilerError::FileCorrupted(f) => {
                println!("Could not read input file `{}`, because the file is corrupted or otherwise not understandable.", f.to_string().bold().underlined())
            }
            CompilerError::VerifyFailed(msg) => {
                print!("Failed to verify produced module, due to a mistake by the compiler:\n{msg}")
            }
        }
    }
}

#[derive(Debug, FromPrimitive)]
pub enum CodeErrorType {
    LexerUnknownChar,
    LexerUnexpectedChar,
    LexerEndOfFile,
    ParserUnexpectedToken,
    MissingTokenError,
    FunctionOverloaded,
    NotAType,
    UnallowedVoid,
    AlreadyExists,
    NotAFunction,
    TypeMismatch,
    FunctionArgumentCount,
    FunctionArgumentType,
    SymbolNotFound,
    VoidReturn,
    InvalidVarDef,
    NonVoidReturn,
    Unable2Cast,
    Uncastable,
    ToplevelStatement,
    BreakOutsideLoop,
    NonVoidNoReturn,
    FieldNotFound,
    WrongDirectiveSignature,
    UnknownDirective,
    BinOpOnNonPrimitiveType,
    IsSigned,
    CanOnlyFreePointers,
    ConditionsMustBeBools,
    ErrorFromDirective,
    WrongFirstMethodParam,
    ReservedName,
    UnknownMemberFunction,
}

#[derive(Debug)]
pub enum CodeWarningType {
    DeadCode,
    UnnecessaryCode,
    DiscouragedPractice,
    DirectiveWarning
}

#[derive(Debug)]
pub struct CodeError {
    pub position: CodePosition,
    pub code_error_type: CodeErrorType,
    pub title: String,
    pub footer: String,
    pub pointer: Option<String>,
    pub notes: Vec<String>,
}

impl CodeError {
    pub fn new(
        position: CodePosition,
        code_error_type: CodeErrorType,
        title: String,
        pointer: Option<String>,
        footer: String,
        notes: Vec<String>,
    ) -> Self {
        Self {
            position,
            code_error_type,
            title,
            footer,
            pointer,
            notes,
        }
    }

    pub fn new_unexpected_token_error(
        token: &Token,
        expected: TokenType,
        extra: Option<String>,
    ) -> Self {
        Self::new(
            token.code_position,
            CodeErrorType::ParserUnexpectedToken,
            "Unexpected Token".to_string(),
            Some(format!("Expected `{}` here", expected)),
            format!(
                "Expected another token `{}`, but got `{}`",
                expected, token.token_type
            ),
            if extra.is_some() {
                vec![extra.unwrap()]
            } else {
                vec![]
            }
        )
    }

    pub fn new_unknown_char_error(position: CodePosition, c: char) -> Self {
        Self::new(
            position,
            CodeErrorType::LexerUnknownChar,
            "Unknown character".to_string(),
            Some("This one".to_string()),
            format!("Character `{}` is weird!", c),
            vec![],
        )
    }

    pub fn void_type(cpos: CodePosition) -> Self {
        Self::new(
            cpos,
            CodeErrorType::UnallowedVoid,
            "Void type is only allowed for functions".to_string(),
            Some("But was used here".to_string()),
            "Use a different type, but not void".to_string(),
            vec![],
        )
    }

    pub fn void_return(cpos: &CodePosition) -> Self {
        Self::new(
            *cpos,
            CodeErrorType::VoidReturn,
            "Void function as must-use-expression".to_string(),
            Some("But was used here".to_string()),
            "A void-typed function can not be used as an expression".to_string(),
            vec![],
        )
    }

    pub fn wrong_first_method_param(tok: &Token, typ: &TypesKind) -> Self {
        Self::new(
            tok.code_position,
            CodeErrorType::WrongFirstMethodParam,
            "Wrong first method parameter".to_string(),
            Some("Concerning this method".to_string()),
            format!("A method must have its first parameter be a pointer to the struct it implements on, but this one has `{typ}`"),
            vec![],
        )
    }

    pub fn unknown_member_function(parent_cpos: CodePosition, stct_name: &String, child: &Token) -> Self {
        Self::new(
            parent_cpos.merge(child.code_position),
            CodeErrorType::UnknownMemberFunction,
            "Unknown member function".to_string(),
            Some("Called here".to_string()),
            format!("The struct `{stct_name}` has no member function `{child}` in its scope"),
            vec![],
        )
    }

    pub fn reserved_name(tok: &Token) -> Self {
        Self::new(
            tok.code_position,
            CodeErrorType::ReservedName,
            "Use of a reserve name".to_string(),
            Some("This name".to_string()),
            format!("Names starting with `__` (like `{tok}`) are reserved and can thus not be used"),
            vec![],
        )
    }

    pub fn invalid_vardef(cpos: CodePosition, t: bool) -> Self {
        let note = (if t {"Add a type"} else {"Add a value"}).to_string();
        Self::new(
            cpos,
            CodeErrorType::InvalidVarDef,
            "Invalid variable definition".to_string(),
            Some("Concerning this variable".to_string()),
            "Creating a variable means that a type or a value must be provided".to_string(),
            vec![note],
        )
    }

    pub fn new_eof_error() -> Self {
        Self::new(
            CodePosition::eof(),
            CodeErrorType::LexerEndOfFile,
            "End of File".to_string(),
            None,
            "Premature end of file!".to_string(),
            vec![]
        )
    }

    pub fn missing_token_error(last_token: &Token) -> Self {
        Self::new(
            last_token.code_position,
            CodeErrorType::MissingTokenError,
            "Missing token".to_string(),
            Some("After this".to_string()),
            "Premature end of file!".to_string(),
            vec![],
        )
    }

    pub fn argument_count(code_position: CodePosition, got: usize, requires: usize) -> Self {
        Self::new(
            code_position,
            CodeErrorType::FunctionArgumentCount,
            "Wrong amount of arguments".to_string(),
            Some("help: adjust the argument count you are passing".to_string()),
            format!("Expected {requires}, but got {got}"),
            vec![]
        )
    }

    pub fn invalid_cast(code_position: CodePosition, new_typ: &TypesKind, expr_typ: &TypesKind) -> Self {
        Self::new(
            code_position,
            CodeErrorType::Unable2Cast,
            "Unable to cast incompatible types".to_string(),
            Some(format!("Cast as type `{new_typ}`")),
            format!("Can not cast an expression of type `{expr_typ}` to `{new_typ}`"),
            vec!["You may not cast a function / struct (either way)".to_string()]
        )
    }

    pub fn type_mismatch(pos: &CodePosition, got: &TypesKind, requires: &TypesKind, notes: Vec<String>) -> Self {
        Self::new(
            *pos,
            CodeErrorType::TypeMismatch,
            "Type mismatch".to_string(),
            Some(format!("This is of type `{got}`")),
            format!("Expected type `{requires}`, but got type `{got}`"),
            notes
        )
    }

    pub fn loop_stmt_outside_loop(pos: &CodePosition, stmt: &TokenType) -> Self {
        Self::new(
            *pos,
            CodeErrorType::BreakOutsideLoop,
            "Loop statement outside of loop".to_string(),
            Some(format!("{stmt} here")),
            "However, break can only be used inside of loops, e.g. while".to_string(),
            vec![]
        )
    }

    pub fn non_void_ret(ret_tok: &Token, name: &String, ret: &TypesKind) -> Self {
        Self::new(
            ret_tok.code_position,
            CodeErrorType::NonVoidReturn,
            "Non void function must return a value".to_string(),
            Some("Return statement with no value".to_string()),
            format!("The function `{name}` should return the type `{ret}`, but here it does not return anything, which is only allowed for void functions"),
            vec![]
        )
    }

    pub fn non_void_no_ret_func(name: &Token, ret: &TypesKind) -> Self {
        Self::new(
            name.code_position,
            CodeErrorType::NonVoidNoReturn,
            "Non void function must return a value".to_string(),
            Some("Function defined here".to_string()),
            format!("The function `{name}` should return the type `{ret}`, but it has no return statement, which is only allowed for void functions"),
            vec![]
        )
    }

    pub fn missing_ends_error(code_position: CodePosition, token_type: TokenType) -> Self {
        Self::new(
            code_position,
            CodeErrorType::ParserUnexpectedToken,
            "Wrong token, expected other".to_string(),
            Some(format!("help: add `{}` here (or after)", token_type)),
            format!("Expected a `{}` at end the statement", token_type),
            vec![]
        )
    }

    pub fn error_from_directive(code_position: CodePosition, msg: String, note: Vec<String>) -> Self {
        Self::new(
            code_position,
            CodeErrorType::ParserUnexpectedToken,
            "Directive triggered an error".to_string(),
            Some("Triggered here".to_string()),
            msg,
            note
        )
    }

    pub fn symbol_not_found(token: &Token) -> Self {
        Self::new(
            token.code_position,
            CodeErrorType::SymbolNotFound,
            "Symbol was not found".to_string(),
            Some("This symbol".to_string()),
            format!("The name `{token}` is referenced here, but can't be resolved"),
            vec![]
        )
    }

    pub fn prim_symbol_not_found(name: &String, code_position: CodePosition) -> Self {
        Self::new(
            code_position,
            CodeErrorType::SymbolNotFound,
            "Symbol was not found".to_string(),
            Some("This symbol".to_string()),
            format!("The name `{name}` is referenced here, but can't be resolved"),
            vec![]
        )
    }

    pub fn prim_module_not_found(name: &String, code_position: CodePosition) -> Self {
        Self::new(
            code_position,
            CodeErrorType::SymbolNotFound,
            "Module was not found".to_string(),
            Some("This module".to_string()),
            format!("The module `{name}` is referenced here, but can't be resolved"),
            vec![format!("You might want to add an import statement: `import {name}`")]
        )
    }

    pub fn field_not_found(token: &Token, struct_name: &String) -> Self {
        Self::new(
            token.code_position,
            CodeErrorType::FieldNotFound,
            "Field not found".to_string(),
            Some("This field".to_string()),
            format!("The field `{token}` of struct {struct_name} is referenced here, but the struct has no such field"),
            vec![]
        )
    }

    pub fn wrong_directive_arg_sig(name: &Token, code_position: CodePosition, expected: Vec<VirtualDirectiveArgType>, got: Vec<VirtualDirectiveArgType>) -> Self {
        Self::new(
            code_position,
            CodeErrorType::WrongDirectiveSignature,
            "Wrong directive signature".to_string(),
            Some(format!("For the directive `{name}`")),
            format!("The directive `{name}` expected the signature {}, but got {}", 
                    format_virtual_type_sig(&name.content.to_uppercase(), expected), format_virtual_type_sig(&name.content.to_uppercase(), got)),
            vec![]
        )
    }

    pub fn unknown_directive(name: &Token) -> Self {
        Self::new(
            name.code_position,
            CodeErrorType::UnknownDirective,
            "Unknown Directive".to_string(),
            Some(format!("Called `{name}`")),
            format!("The directive `{name}` does not exist"),
            vec!["Did you spell it right?".to_string(), "Consult the README for a list of directives".to_string()]
        )
    }

    pub fn bin_op_on_non_primitive_type(code_position: CodePosition, typ: TypesKind) -> Self {
        Self::new(
            code_position,
            CodeErrorType::BinOpOnNonPrimitiveType,
            "Binary operation on non primitive type".to_string(),
            Some(format!("This is of type `{typ}`")),
            "Can only do binary operations on primitive types".to_string(),
            vec!["Primitive types are: i32, i64, u32, u64, u8, bool, f32, f64".to_string()]
        )
    }

    pub fn is_signed(code_position: CodePosition, typ: TypesKind) -> Self {
        Self::new(
            code_position,
            CodeErrorType::IsSigned,
            "Value is signed".to_string(),
            Some(format!("This is of type `{typ}` (signed)")),
            "Malloc requires a non-zero value, e.g. an unsinged integer".to_string(),
            vec![]
        )
    }

    pub fn can_only_free_pointers(code_position: CodePosition, typ: TypesKind) -> Self {
        Self::new(
            code_position,
            CodeErrorType::CanOnlyFreePointers,
            "Can only free pointers".to_string(),
            Some(format!("This is of type `{typ}` (not a pointer)")),
            "Free requires a pointer".to_string(),
            vec![]
        )
    }

    pub fn conditions_must_be_bool(code_position: CodePosition, typ: TypesKind) -> Self {
        Self::new(
            code_position,
            CodeErrorType::ConditionsMustBeBools,
            "Conditions must be bools".to_string(),
            Some(format!("This is of type `{typ}` (not a bool)")),
            "Conditions may only be bools".to_string(),
            vec!["You can probably cast your type into a bool".to_string()]
        )
    }

    pub fn symbol_not_a_function(mav: &ModuleAccessVariant) -> Self {
        Self::new(
            mav.ensured_compute_codeposition(),
            CodeErrorType::NotAFunction,
            "Symbol is not a function".to_string(),
            Some("Called here".to_string()),
            format!("The name `{mav}` is found, but not a symbol"),
            vec![]
        )
    }

    pub fn toplevel_statement(cpos: CodePosition) -> Self {
        Self::new(
            cpos,
            CodeErrorType::ToplevelStatement,
            "Not a top level statement".to_string(),
            Some("here".to_string()),
            "This kind of statement is only allowed in a function".to_string(),
            vec![
                "Only function definitions and imports are top level statements".to_string(),
                "Maybe you forgot to put it into a function?".to_string()]
        )
    }

    pub fn not_a_type_error(token: &Token) -> Self {
        Self::new(token.code_position, CodeErrorType::NotAType, "Wrong token, expected a type".to_string(), None, format!("Expected a valid type, not `{token}`"), vec![])
    }

    pub fn function_overloaded(already_token: &Token) -> Self {
        Self::new(
            already_token.code_position,
            CodeErrorType::FunctionOverloaded,
            "Function was overloaded".to_string(),
            Some("Function mode set here".to_string()),
            "Can not have multiple function modes".to_string(),
            vec!["Remove one of the modifiers".to_string()],
        )
    }

    pub fn already_exists(f: bool, symbol: &Token) -> Self {
        Self::new(
            symbol.code_position,
            CodeErrorType::AlreadyExists,
            "A symbol with the same name already exists".to_string(),
            Some("This".to_string()),
            "Symbols must be unique".to_string(),
            vec!["Remove one of the them".to_string()],
        )
    }

    pub fn visualize_error(self, file_manager: &FileManager) {
        print_code_error(self, file_manager)
    }
}

pub type CompResult<T> = Result<T, CompilerError>;
pub type CodeResult<T> = Result<T, CodeError>;

impl fmt::Display for CodeErrorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct CodeWarning {
    pub position: CodePosition,
    pub code_warn_type: CodeWarningType,
    pub title: String,
    pub footer: String,
    pub pointer: Option<String>,
    pub notes: Vec<String>,
}

impl CodeWarning {
    pub fn new(
        position: CodePosition,
        code_warn_type: CodeWarningType,
        title: String,
        pointer: Option<String>,
        footer: String,
        notes: Vec<String>,
    ) -> Self {
        Self {
            position,
            code_warn_type,
            title,
            footer,
            pointer,
            notes,
        }
    }

    pub fn new_unnecessary_code(position: CodePosition, extra: Option<String>) -> Self {
        Self::new(
            position,
            CodeWarningType::UnnecessaryCode,
            "Unnecessary code".to_string(),
            None,
            "This code does not change the outcome".to_string(),
            if extra.is_some() {
                vec![extra.unwrap()]
            } else {
                vec!["You should remove it".to_string()]
            },
        )
    }

    pub fn dead_code(position: CodePosition, extra: Option<String>) -> Self {
        Self::new(
            position,
            CodeWarningType::DeadCode,
            "Dead code".to_string(),
            None,
            "This code is unreachable".to_string(),
            if extra.is_some() {
                vec![extra.unwrap()]
            } else {
                vec!["You should remove it".to_string()]
            },
        )
    }

    pub fn directive_warning(position: CodePosition, msg: String) -> Self {
        Self::new(
            position,
            CodeWarningType::DirectiveWarning,
            "Warning via directive".to_string(),
            Some("Triggered here".to_string()),
            msg,
            vec![]
        )
    }
}
