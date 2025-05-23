use std::collections::HashMap;
use std::ops::Deref;
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder};
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, FloatType, FunctionType, IntType};
use inkwell::values::{AsValueRef, BasicValue, FloatValue, FunctionValue, InstructionOpcode, IntMathValue, IntValue, PointerValue};
use crate::codeviz::print_code_warn;
use crate::comp_errors::{CodeError, CodeResult, CodeWarning};
use crate::filemanager::FileManager;
use crate::lexer::{CodePosition, Token};
use crate::parser::{BinaryOp, Expression, ExpressionKind, FunctionMode, Types, TypesKind, AST};

enum PrimitiveErrors {
    TypeVoidUnallowed
}

type PrimRes<T> = Result<T, PrimitiveErrors>;

fn resolve_prim_res<'a>(result: Result<FunctionType<'a>, PrimitiveErrors>, tok: &Token) -> CodeResult<FunctionType<'a>> {
    result.map_err(|_| {CodeError::void_type(tok)})
}

fn is_type_signed(typ: &TypesKind) -> bool {
    match typ {
        TypesKind::U32 | TypesKind::U64 | TypesKind::U8 => true,
        _ => false
    }
}

fn hinted_int<'a>(hint: Option<&TypesKind>, ctx: &'a Context) -> (IntType<'a>, TypesKind) {
    match hint {
        Some(h) => {
            match h {
                TypesKind::I64 | TypesKind::U64 => (ctx.i64_type(), h.to_owned()),
                TypesKind::U8 => (ctx.i8_type(), h.to_owned()),
                TypesKind::U32 => (ctx.i32_type(), h.to_owned()),
                _ => (ctx.i32_type(), TypesKind::I32),
            }
        } None => (ctx.i32_type(), TypesKind::I32)
    }
}

pub fn hinted_float<'a>(hint: Option<&TypesKind>, ctx: &'a Context) -> (FloatType<'a>, TypesKind) {
    match hint {
        Some(h) => {
            match h {
                TypesKind::F64 => (ctx.f64_type(), TypesKind::F64),
                _ => (ctx.f32_type(), TypesKind::F32),
            }
        } None => (ctx.f32_type(), TypesKind::F32)
    }
}

pub struct Function<'ctx> {
    compiler: &'ctx Compiler<'ctx>,
    function_value: FunctionValue<'ctx>,
    ret: &'ctx TypesKind,
    function_scope: Namespace<'ctx>,
    name: String,
}

impl<'ctx> Function<'ctx> {
    pub fn new(compiler: &'ctx Compiler<'ctx>, function_value: FunctionValue<'ctx>, ret: &'ctx TypesKind, name: String) -> Self {
        Self { compiler, function_value, ret, function_scope: Namespace::new(), name }
    }

    pub fn new_block(&self, block_name: &str, pae: bool) -> BasicBlock<'ctx> {
        let bb = self.compiler.context.append_basic_block(self.function_value, block_name);
        if pae { self.compiler.builder.position_at_end(bb) }
        bb
    }
}

pub struct Namespace<'ns> {
    pub definitions: HashMap<String, (TypesKind, PointerValue<'ns>)>,
    pub functions: HashMap<String, (FunctionValue<'ns>, bool)>
}

// Check local and global scope
fn check_ls_gs_defs<'a>(name: &str, ls: &'a Namespace<'a>, gs: &'a Namespace<'a>) -> Option<&'a (TypesKind, PointerValue<'a>)> {
    ls.definitions.get(name).or(gs.definitions.get(name))
}
fn check_ls_gs_func<'a>(name: &str, ls: &'a Namespace<'a>, gs: &'a Namespace<'a>) -> Option<&'a (FunctionValue<'a>, bool)> {
    ls.functions.get(name).or(gs.functions.get(name))
}

impl<'ns> Namespace<'ns> {
    fn new() -> Self {
        Self { definitions: HashMap::new(), functions: HashMap::new() }
    }
}

fn basic_value_box_into_int<'a>(value: Box<(dyn BasicValue<'a> + 'a)>) -> IntValue<'a> {
    value.as_basic_value_enum().into_int_value()
}

fn basic_value_box_into_float<'a>(value: Box<(dyn BasicValue<'a> + 'a)>) -> FloatValue<'a> {
    value.as_basic_value_enum().into_float_value()
}

macro_rules! build_bin_op {
    ($builder:expr, $kind:ident, $lhs:expr, $rhs:expr, $op:ident, $expect:expr) => {
        Box::new($builder.$kind($lhs, $rhs, "").expect($expect))
    };
}


pub struct Compiler<'ctx> {
    context: &'ctx Context,
    builder: &'ctx Builder<'ctx>,
    module_name: String,
    file_manager: &'ctx FileManager
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(context: &'ctx Context, builder: &'ctx Builder<'ctx>, module_name: String, file_manager: &'ctx FileManager) -> Self {
        Self { context, builder, module_name, file_manager }
    }

    fn convert_type_normal<'a>(&'a self, typ: &TypesKind) -> PrimRes<Box<dyn BasicType + 'a>> {
        match typ {
            TypesKind::I32 | TypesKind::U32 => Ok(Box::new(self.context.i32_type())),
            TypesKind::F32 => Ok(Box::new(self.context.f32_type())),
            TypesKind::Void => Err(PrimitiveErrors::TypeVoidUnallowed),
            TypesKind::Struct { .. } => todo!("Add structs"),
            TypesKind::Function { ret, params } => {
                Ok(Box::new(self.convert_type_function(&*ret.to_owned(), params.iter().map(|x| {x.to_owned()}).collect())?.ptr_type(AddressSpace::default())))
            },
            TypesKind::Ptr(ptr) => {
                Ok(Box::new(self.convert_type_normal(ptr)?.ptr_type(AddressSpace::default()).as_basic_type_enum()))
            }
            TypesKind::Pointer => Ok(Box::new(self.context.ptr_type(AddressSpace::default()).as_basic_type_enum())),
            TypesKind::I64 | TypesKind::U64 => Ok(Box::new(self.context.i32_type())),
            TypesKind::F64  => Ok(Box::new(self.context.f64_type())),
            TypesKind::U8 => Ok(Box::new(self.context.i8_type())),
            TypesKind::Bool => Ok(Box::new(self.context.custom_width_int_type(1).as_basic_type_enum())),
        }
    }

    fn convert_type_function(&self, ret_type: &TypesKind, params: Vec<TypesKind>) -> PrimRes<FunctionType> {
        let param_types: Vec<BasicMetadataTypeEnum> = params
            .iter()
            .map(|typ| {
                let converted = self.convert_type_normal(typ)?; // This returns a Result
                Ok(BasicMetadataTypeEnum::from(converted.as_basic_type_enum()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        match ret_type {
            TypesKind::I32 | TypesKind::U32 => Ok(self.context.i32_type().fn_type(&param_types, false)),
            TypesKind::F32 => Ok(self.context.f32_type().fn_type(&param_types, false)),
            TypesKind::Void => Ok(self.context.void_type().fn_type(&param_types, false)),
            TypesKind::Struct { .. } => todo!("Add structs"),
            TypesKind::Function { ret,params } => {
                Ok(*Box::new(self.convert_type_function(&*ret.to_owned(), params.iter().map(|x| { x.to_owned() }).collect())?.ptr_type(AddressSpace::default()).fn_type(&param_types, false)))
            },
            TypesKind::Ptr(ptr) => {
                Ok(*Box::new(self.convert_type_normal(ptr)?.ptr_type(AddressSpace::default()).fn_type(&param_types, false)))
            },
            TypesKind::Pointer => Ok(*Box::new(self.context.ptr_type(AddressSpace::default()).fn_type(&param_types, false))),
            TypesKind::I64 | TypesKind::U64 => Ok(self.context.i64_type().fn_type(&param_types, false)),
            TypesKind::F64 => Ok(self.context.f64_type().fn_type(&param_types, false)),
            TypesKind::U8 => Ok(self.context.i8_type().fn_type(&param_types, false)),
            TypesKind::Bool => Ok(self.context.custom_width_int_type(1).fn_type(&param_types, false)),
        }
    }
    
    fn to_bool_int<'a>(&'a self, cond: IntValue<'a>) -> Box<dyn BasicValue + 'a> {
        Box::new(self.builder.build_int_compare(IntPredicate::NE, cond, self.context.custom_width_int_type(1).const_zero(), "is_nonzero").expect("Failed cond comp").as_basic_value_enum())
    }

    fn visit_bin_op<'a>(&'a self, lhs: Box<(dyn BasicValue<'a> + 'a)>, rhs: Box<(dyn BasicValue<'a> + 'a)>, op: BinaryOp, typ: TypesKind) -> CodeResult<(Box<dyn BasicValue + 'a>, TypesKind)> {
        let result: (Box<dyn BasicValue<'a>>, TypesKind) = match typ {
            TypesKind::I32 | TypesKind::U32 | TypesKind::I64 | TypesKind::U64 | TypesKind::U8 => {
                let lhs = basic_value_box_into_int(lhs);
                let rhs = basic_value_box_into_int(rhs);

                let build_cmp = |pred| self.to_bool_int(self.builder.build_int_compare(pred, lhs, rhs, "").expect("Int comparison failed"));

                match op {
                    BinaryOp::Eq  => (Box::new(build_cmp(IntPredicate::EQ).as_basic_value_enum()), TypesKind::Bool),
                    BinaryOp::Neq => (Box::new(build_cmp(IntPredicate::NE).as_basic_value_enum()), TypesKind::Bool),
                    BinaryOp::Gt  => (Box::new(build_cmp(IntPredicate::UGT).as_basic_value_enum()), TypesKind::Bool),
                    BinaryOp::Gte => (Box::new(build_cmp(IntPredicate::UGE).as_basic_value_enum()), TypesKind::Bool),
                    BinaryOp::Lt  => (Box::new(build_cmp(IntPredicate::ULT).as_basic_value_enum()), TypesKind::Bool),
                    BinaryOp::Lte => (Box::new(build_cmp(IntPredicate::ULE).as_basic_value_enum()), TypesKind::Bool),
                    BinaryOp::And => (Box::new(self.builder.build_and(lhs, rhs, "").expect("Failed int op").as_basic_value_enum()), TypesKind::Bool),
                    BinaryOp::Or  => (Box::new(self.builder.build_or(lhs, rhs, "").expect("Failed int op").as_basic_value_enum()), TypesKind::Bool),
                    
                    BinaryOp::Add => (Box::new(self.builder.build_int_add(lhs, rhs, "").expect("Failed int op").as_basic_value_enum()), typ),
                    BinaryOp::Sub => (Box::new(self.builder.build_int_sub(lhs, rhs, "").expect("Failed int op").as_basic_value_enum()), typ),
                    BinaryOp::Mul => (Box::new(self.builder.build_int_mul(lhs, rhs, "").expect("Failed int op").as_basic_value_enum()), typ),
                    BinaryOp::Div => (Box::new(self.builder.build_int_unsigned_div(lhs, rhs, "").expect("Failed int op").as_basic_value_enum()), typ),
                }
            }

            TypesKind::F32 | TypesKind::F64 => {
                let lhs = basic_value_box_into_float(lhs);
                let rhs = basic_value_box_into_float(rhs);

                let build_cmp = |pred| self.to_bool_int(self.builder.build_float_compare(pred, lhs, rhs, "").expect("Float comparison failed"));

                match op {
                    BinaryOp::Eq  => (Box::new(build_cmp(FloatPredicate::OEQ).as_basic_value_enum()), TypesKind::Bool),
                    BinaryOp::Neq => (Box::new(build_cmp(FloatPredicate::ONE).as_basic_value_enum()), TypesKind::Bool),
                    BinaryOp::Gt  => (Box::new(build_cmp(FloatPredicate::OGT).as_basic_value_enum()), TypesKind::Bool),
                    BinaryOp::Gte => (Box::new(build_cmp(FloatPredicate::OGE).as_basic_value_enum()), TypesKind::Bool),
                    BinaryOp::Lt  => (Box::new(build_cmp(FloatPredicate::OLT).as_basic_value_enum()), TypesKind::Bool),
                    BinaryOp::Lte => (Box::new(build_cmp(FloatPredicate::OLE).as_basic_value_enum()), TypesKind::Bool),
                    BinaryOp::And | BinaryOp::Or => {
                        let int_ty = self.context.i32_type();
                        let lhs_int = self.builder.build_cast(InstructionOpcode::FPToSI, lhs, int_ty, "")
                            .expect("Cast float to int lhs").into_int_value();
                        let rhs_int = self.builder.build_cast(InstructionOpcode::FPToSI, rhs, int_ty, "")
                            .expect("Cast float to int rhs").into_int_value();
                        let logic_op = match op {
                            BinaryOp::And => self.builder.build_and(lhs_int, rhs_int, ""),
                            BinaryOp::Or  => self.builder.build_or(lhs_int, rhs_int, ""),
                            _ => unreachable!(),
                        };
                        (Box::new(logic_op.expect("Float logic failed")), TypesKind::Bool)
                    }
                    BinaryOp::Add => (Box::new(self.builder.build_float_add(lhs, rhs, "").expect("Failed float op").as_basic_value_enum()), typ),
                    BinaryOp::Sub => (Box::new(self.builder.build_float_sub(lhs, rhs, "").expect("Failed float op").as_basic_value_enum()), typ),
                    BinaryOp::Mul => (Box::new(self.builder.build_float_mul(lhs, rhs, "").expect("Failed float op").as_basic_value_enum()), typ),
                    BinaryOp::Div => (Box::new(self.builder.build_float_div(lhs, rhs, "").expect("Failed float op").as_basic_value_enum()), typ),
                }
            }

            _ => panic!(""),
        };
        Ok(result)
    }
    
    fn null(&self) -> IntValue<'ctx> {
        self.context.i32_type().const_zero()
    }
    

    fn visit_expr<'a>(&'a self, function: &'a Function<'a>, global_scope: &'a Namespace<'_>, expr: Expression, 
                      type_hint: Option<&TypesKind>, must_use: bool) -> CodeResult<(Box<dyn BasicValue + 'a>, TypesKind)> {
        Ok(match expr.expression {
            ExpressionKind::Identifier(name) => {
                let def = check_ls_gs_defs(&name.content, &function.function_scope, global_scope).ok_or_else(|| {CodeError::symbol_not_found(name)})?;
                let real_type = self.convert_type_normal(&def.0).map_err(|e| {CodeError::void_type(name)})?;
                (Box::new(self.builder.build_load(real_type.as_basic_type_enum(), def.1, &name.content).expect("Failed to load")), def.0.clone())
            }
            ExpressionKind::IntNumber { value, .. } => {
                let (hint, vt) = hinted_int(type_hint, self.context);
                (Box::new(hint.const_int(value, is_type_signed(&vt)).as_basic_value_enum()), vt)
            }
            ExpressionKind::FloatNumber { value, .. } => {
                let(hint, vt) = hinted_float(type_hint, self.context);
                (Box::new(hint.const_float(value).as_basic_value_enum()), vt)
            }
            ExpressionKind::BinaryOp { lhs, op, rhs } => {
                let cpos = rhs.code_position;
                let (lhs_val, l_typ) = self.visit_expr(function, global_scope, *lhs, type_hint, true)?;
                let (rhs_val, r_typ) = self.visit_expr(function, global_scope, *rhs, Some(&l_typ), true)?;
                if l_typ != r_typ {
                    return Err(CodeError::type_mismatch(&cpos, &r_typ, &l_typ, vec!["This is because the right side must have the same type as the left side".to_string()]));
                }
                self.visit_bin_op(lhs_val, rhs_val, op.1, l_typ)?
            }
            ExpressionKind::FunctionCall { name, arguments } => {
                let def = check_ls_gs_defs(&name.content, &function.function_scope, global_scope).ok_or_else(|| {
                    CodeError::symbol_not_found(name)
                })?;
                match &def.0 {
                    TypesKind::Function { params, ret } => {
                        if let Some(hint) = type_hint {
                            if must_use && *hint == TypesKind::Void {
                                return Err(CodeError::void_return(&expr.code_position));
                            }
                        }
                        let mut h_args = vec![];
                        if arguments.len() != params.len() {
                            return Err(CodeError::argument_count(name, arguments.len(), params.len()))
                        }
                        for (count, arg) in arguments.into_iter().enumerate() {
                            let cpos = arg.code_position;
                            let (value, typ) = self.visit_expr(function, global_scope, arg, Some(&params[count]), true)?;
                            if typ != params[count] {
                                return Err(CodeError::type_mismatch(&cpos, &typ, &params[count], vec![]));
                            }
                            h_args.push(value.deref().as_basic_value_enum().into());
                        }
                        // Call function, if it does not return a value (Void function) return null
                        // In that case the return value will NOT be used, because of other Void checks
                        let call = self.builder.build_call(check_ls_gs_func(&name.content, &function.function_scope, global_scope)
                            .expect("Failed to get function - which should exist btw").0, h_args.as_slice(), "").expect("Failed to load");
                        (Box::new(call.try_as_basic_value().left_or(self.null().as_basic_value_enum())), ret.as_ref().clone())
                    }
                    _ => Err(CodeError::symbol_not_a_function(name))?
                }
            },
            ExpressionKind::String(string) => {
                let global = self.builder.build_global_string_ptr(&string.content, "").expect("Failed to create global str ptr").as_pointer_value();
                (Box::new(global.as_basic_value_enum()), TypesKind::Ptr(Box::new(TypesKind::U8)))
            }
            ExpressionKind::Type { .. } => {todo!("Implement")}
            ExpressionKind::CastExpr { expr, typ: new_type } => {
                let (value, old_type) = self.visit_expr(function, global_scope, *expr, None, true)?;
                let value = value.as_basic_value_enum();
                let real_new_typ = self.convert_type_normal(&new_type.kind)
                    .map_err(|_| CodeError::void_type(new_type.token))?
                    .as_basic_type_enum();

                let result = match old_type {
                    TypesKind::I32 | TypesKind::U32 | TypesKind::U8 | TypesKind::I64 | TypesKind::U64 | TypesKind::Bool => {
                        let int_val = value.into_int_value();
                        match &new_type.kind {
                            TypesKind::I32 | TypesKind::U32 | TypesKind::U8 | TypesKind::I64 | TypesKind::U64 | TypesKind::Bool => {
                                let dest_type = real_new_typ.into_int_type();
                                if int_val.get_type().get_bit_width() < dest_type.get_bit_width() {
                                    self.builder
                                        .build_int_z_extend(int_val, dest_type, "")
                                        .map(|v| v.as_basic_value_enum())
                                        .ok()
                                } else if int_val.get_type().get_bit_width() > dest_type.get_bit_width() {
                                    self.builder
                                        .build_int_truncate(int_val, dest_type, "")
                                        .map(|v| v.as_basic_value_enum())
                                        .ok()
                                } else {
                                    return Ok((Box::new(value), old_type));
                                }
                            }
                            TypesKind::F32 | TypesKind::F64 => {
                                self.builder
                                    .build_unsigned_int_to_float(
                                        int_val,
                                        real_new_typ.into_float_type(),
                                        "",
                                    )
                                    .map(|v| v.as_basic_value_enum())
                                    .ok()
                            }
                            TypesKind::Void => None,
                            TypesKind::Ptr(_) | TypesKind::Pointer => {
                                self.builder
                                    .build_int_to_ptr(int_val, real_new_typ.into_pointer_type(), "")
                                    .map(|v| v.as_basic_value_enum())
                                    .ok()
                            }
                            _ => None,
                        }
                    }
                    TypesKind::F32 | TypesKind::F64 => {
                        let float_val = value.into_float_value();
                        match &new_type.kind {
                            TypesKind::F32 | TypesKind::F64 => {
                                let dest_type = real_new_typ.into_float_type();
                                if float_val.get_type().size_of().get_sign_extended_constant().unwrap() < dest_type.size_of().get_sign_extended_constant().unwrap() {
                                    self.builder
                                        .build_float_ext(float_val, dest_type, "")
                                        .map(|v| v.as_basic_value_enum())
                                        .ok()
                                } else if float_val.get_type().size_of().get_sign_extended_constant().unwrap() > dest_type.size_of().get_sign_extended_constant().unwrap() {
                                    self.builder
                                        .build_float_trunc(float_val, dest_type, "")
                                        .map(|v| v.as_basic_value_enum())
                                        .ok()
                                } else {
                                    return Ok((Box::new(value), old_type));
                                }
                            }
                            TypesKind::I32 | TypesKind::I64 | TypesKind::U32 | TypesKind::U64 | TypesKind::U8 => {
                                self.builder
                                    .build_float_to_unsigned_int(
                                        float_val,
                                        real_new_typ.into_int_type(),
                                        "",
                                    )
                                    .map(|v| v.as_basic_value_enum())
                                    .ok()
                            }
                            _ => None,
                        }
                    }
                    TypesKind::Ptr(_) | TypesKind::Pointer => {
                        let ptr_val = value.into_pointer_value();
                        match &new_type.kind {
                            TypesKind::I32 | TypesKind::I64 | TypesKind::U32 | TypesKind::U64 => {
                                self.builder
                                    .build_ptr_to_int(ptr_val, real_new_typ.into_int_type(), "")
                                    .map(|v| v.as_basic_value_enum())
                                    .ok()
                            }
                            TypesKind::Ptr(ptr_type) => {
                                let target_type = self.convert_type_normal(ptr_type)
                                    .map_err(|_| CodeError::void_type(new_type.token))?
                                    .ptr_type(AddressSpace::default());
                                self.builder
                                    .build_pointer_cast(ptr_val, target_type, "")
                                    .map(|v| v.as_basic_value_enum())
                                    .ok()
                            }
                            TypesKind::Pointer => {
                                let target_type = self.context.ptr_type(AddressSpace::default());
                                self.builder
                                    .build_pointer_cast(ptr_val, target_type, "")
                                    .map(|v| v.as_basic_value_enum())
                                    .ok()
                            }
                            _ => None,
                        }
                    }
                    TypesKind::Void | TypesKind::Struct { .. } | TypesKind::Function { .. } => None,
                };

                (Box::new(result.ok_or_else(|| {
                    CodeError::invalid_cast(new_type.token, &new_type.kind, &old_type)
                })?), new_type.kind)
            }
            ExpressionKind::Reference { var } => {
                let def = check_ls_gs_defs(&var.content, &function.function_scope, global_scope).ok_or_else(|| {
                    CodeError::symbol_not_found(var)
                })?;
                (Box::new(def.1.as_basic_value_enum()), TypesKind::Ptr(Box::new(def.0.clone())))
            }
            ExpressionKind::Dereference { var } => {
                let def = check_ls_gs_defs(&var.content, &function.function_scope, global_scope).ok_or_else(|| {
                    CodeError::symbol_not_found(var)
                })?;
                
                let deref_type = match &def.0 {
                    TypesKind::Ptr(ptr_t) => {ptr_t},
                    TypesKind::Pointer => &{ Box::new(TypesKind::Void) },
                    _ => {
                        return Err(CodeError::type_mismatch(&var.code_position, &def.0, &TypesKind::Pointer, vec!["Only pointers can be dereferenced, but this variable is not a pointer".to_string()]))
                    }
                };

                let ptr_t = self.convert_type_normal(&def.0).map_err(|e| {CodeError::void_type(var)})?;
                let real_deref_type = self.convert_type_normal(deref_type).map_err(|e| {CodeError::void_type(var)})?;
                
                let ptr = self.builder.build_load(ptr_t.as_basic_type_enum(), def.1, "")
                    .expect("Load (deref) failed").as_basic_value_enum().into_pointer_value();
                let value = self.builder.build_load(real_deref_type.as_basic_type_enum(), ptr, "")
                    .expect("Load (deref) failed").as_basic_value_enum();
                (Box::new(value), *deref_type.clone())
            }
        })
    }

    fn visit_statement<'a>(&'ctx self, function: &mut Function<'ctx>, statement: AST, global_scope: &mut Namespace<'a>, after_block: Option<BasicBlock>) -> CodeResult<(bool, bool)> {
        let mut inside_loop = after_block.is_some();
        let mut returns = false;
        // Returns: inside_loop, returns
        
        match statement {
            AST::FunctionDef { .. } => panic!("Function definition is not allowed here (YET)"),
            AST::VariableDef { name, value, typ } => {
                let val = if let Some(value) = value {
                    let cpos = &value.code_position.clone();
                    let (expr, typ) = if let Some(typ) = typ {
                        let (expr, got_typ) = self.visit_expr(function, global_scope, value, Some(&typ.kind), true)?;
                        if got_typ != typ.kind { return Err(CodeError::type_mismatch(cpos, &got_typ, &typ.kind, vec![])) }
                        (expr, got_typ)
                    } else {
                        self.visit_expr(function, global_scope, value, None, true)?
                    };
                    let pointer = self.builder.build_alloca(self.convert_type_normal(&typ)
                    .map_err(|_| {CodeError::void_type(name)})?.as_basic_type_enum(), "").expect("Can not allocate for var-define");
                    self.builder.build_store(pointer, expr.as_basic_value_enum()).expect("Failed to store value of variable define");
                    (typ, pointer)
                } else {
                    let typ = typ.unwrap();
                    (typ.kind.clone(), self.builder.build_alloca(self.convert_type_normal(&typ.kind)
                    .map_err(|_| {CodeError::void_type(name)})?.as_basic_type_enum(), "").expect("Can not allocate for var-declare"))
                };
                function.function_scope.definitions.insert(name.content.clone(), val);
            }
            AST::VariableReassign { name, value } => {
                let cpos = &value.code_position.clone();
                let (typ, ptr) = function.function_scope.definitions.get(&name.content).ok_or_else(|| {CodeError::symbol_not_found(name)})?;
                let (data, got_type) = self.visit_expr(function, global_scope, value, Some(typ), true)?;
                if got_type != *typ { return Err(CodeError::type_mismatch(cpos, &got_type, typ, vec![format!("This is because `{}` was originally declared as {typ}", name.content)])) } 
                self.builder.build_store(*ptr, data.as_basic_value_enum()).expect("Failed to store value of variable define");
            }
            AST::Expression { expr } => { 
                self.visit_expr(function, global_scope, expr, None, false)?;
            }
            AST::Return(value, ret_tok) => {
                if let Some(expr) = value {
                    let cpos = &expr.code_position.clone();
                    let (n_value, typ) = self.visit_expr(function, global_scope, expr, Some(function.ret), true)?;
                    if typ != *function.ret {
                        return Err(CodeError::type_mismatch(cpos, &typ, function.ret, vec![format!("This is because the function `{}` has the return-type `{}`", function.name, function.ret)]))
                    }
                    self.builder.build_return(Some(n_value.deref())).expect("Failed to return");
                } else {
                    if *function.ret != TypesKind::Void {
                        return Err(CodeError::non_void_ret(ret_tok, &function.name, function.ret))
                    }
                    self.builder.build_return(None).expect("Failed to return NONE");
                }
                returns = true;
            },
            AST::CondLoop(cond) => {
                inside_loop = true;
                
                let cond_block = function.new_block("cond", false);
                let body_block = function.new_block("body", false); // <- Added body block
                let after_block = function.new_block("after", false);

                self.builder.build_unconditional_branch(cond_block).expect("Failed to enter cond block");

                self.builder.position_at_end(cond_block);
                
                {
                    let (cond_value, typ) = self.visit_expr(function, global_scope, cond.condition, None, true)?;
                    if typ != TypesKind::Bool {
                        panic!("Must be a bool")
                    }
                    let condition = cond_value.as_basic_value_enum().into_int_value();
                    self.builder.build_conditional_branch(condition, body_block, after_block)
                        .expect("Failed to make conditional branch");
                }

                self.builder.position_at_end(body_block);
                for stmt in cond.body {
                    let (il, rets) = self.visit_statement(function, stmt, global_scope, Some(after_block))?;
                    if !il || rets {
                        inside_loop = false;
                        break;
                    }
                }
                self.builder.build_unconditional_branch(cond_block)
                    .expect("Failed to branch back to condition");

                self.builder.position_at_end(after_block);
                return Ok((inside_loop, false));
            }
            AST::IfCondition { first, other, elif } => {
                let after_block = function.new_block("if_after", false);

                let mut cond_blocks = vec![(&first, function.new_block("if_cond0", false))];
                let mut body_blocks = vec![function.new_block("if_body0", false)];

                for (i, elif_block) in elif.iter().enumerate() {
                    cond_blocks.push((elif_block, function.new_block(&format!("elif_cond{i}"), false)));
                    body_blocks.push(function.new_block(&format!("elif_body{i}"), false));
                }

                let else_bb = if other.as_ref().map_or(false, |v| !v.is_empty()) {
                    Some(function.new_block("else_body", false))
                } else {
                    None
                };

                self.builder
                    .build_unconditional_branch(cond_blocks[0].1)
                    .expect("Failed to do jump (IF)");

                for i in 0..cond_blocks.len() {
                    let (cond_block, cond_bb) = cond_blocks[i];
                    let body_bb = body_blocks[i];

                    self.builder.position_at_end(cond_bb);

                    let cond_val = self.visit_expr(
                        function,
                        global_scope,
                        cond_block.condition.clone(),
                        None,
                        true,
                    )?.0.as_any_value_enum().into_int_value();

                    let zero = cond_val.get_type().const_zero();
                    let cond = self.builder
                        .build_int_compare(IntPredicate::NE, cond_val, zero, "if_cond")
                        .expect("Failed to build int-cmp");

                    let next_cond_bb = cond_blocks.get(i + 1).map(|(_, bb)| *bb)
                        .or(else_bb)
                        .unwrap_or(after_block);

                    self.builder
                        .build_conditional_branch(cond, body_bb, next_cond_bb)
                        .expect("Failed to build jump (IF2)");
                }

                for (i, cond_block) in cond_blocks.iter().enumerate() {
                    let body_bb = body_blocks[i];
                    self.builder.position_at_end(body_bb);

                    for stmt in &cond_block.0.body {
                        let continue_block = self.visit_statement(
                            function,
                            stmt.clone(),
                            global_scope,
                            Some(after_block),
                        )?;
                        returns = continue_block.1;
                        if !continue_block.0 || continue_block.1 {
                            break;
                        }
                    }

                    if !returns {
                        self.builder
                            .build_unconditional_branch(after_block)
                            .expect("Failed to build jump (IF4)");
                    }
                }

                if let Some(ref else_stmts) = other {
                    if let Some(else_bb) = else_bb {
                        self.builder.position_at_end(else_bb);

                        for stmt in else_stmts {
                            let continue_block = self.visit_statement(
                                function,
                                stmt.clone(),
                                global_scope,
                                Some(after_block),
                            )?;
                            returns = continue_block.1;
                            if !continue_block.0 || continue_block.1 {
                                break;
                            }
                        }

                        if !returns {
                            self.builder
                                .build_unconditional_branch(after_block)
                                .expect("Failed to build jump (IF4)");
                        }
                    }
                }

                self.builder.position_at_end(after_block);
                
                if other.is_some() && returns {
                    self.builder.build_unreachable().expect("Failed to make after unreachable");
                }
            }
            AST::Break(tok) => {
                if let Some(after) = after_block {
                    self.builder.position_at_end(after);
                    inside_loop = false;
                } else {
                    return Err(CodeError::break_outside_loop(&tok.code_position))
                }
            },
        }
        Ok((inside_loop, returns))
    }
    
    fn ast_to_codepos(ast: &AST) -> CodePosition {
        match ast {
            AST::Expression { expr } => expr.code_position,
            AST::FunctionDef { name, .. } => name.code_position,
            AST::VariableDef { name, .. } => name.code_position,
            AST::VariableReassign { name, .. } => name.code_position,
            AST::Return(_, tok) => tok.code_position,
            AST::IfCondition { first, .. } => first.condition.code_position,
            AST::CondLoop(c) => c.condition.code_position,
            AST::Break(tok) => tok.code_position,
        }
    }

    pub(crate) fn visit_function_def<'a>(&'a self, module: &Module<'a>, name: &Token, fmode: FunctionMode, 
                                         return_type: Types, params: Vec<(&Token, Types)>, global_scope: &mut Namespace<'ctx>,
                                         body: Option<Vec<AST>>) -> CodeResult<()>
    where 'a: 'ctx 
    {
        let fn_type = resolve_prim_res(self.convert_type_function(&return_type.kind, params.iter().map(|x| {x.1.kind.clone()}).collect()), name)?;
        
        if let Some((_, already_defined)) = global_scope.functions.get(&name.content) { 
            if *already_defined {
                return Err(CodeError::already_exists(true, name))
            }
        }
        
        let function_value = module.add_function(&name.content, fn_type, Some(fmode.into()));
        let mut function = Function::new(self, function_value, &return_type.kind, name.content.clone());

        global_scope.functions.insert(name.content.clone(), (function_value, body.is_some()));
        global_scope.definitions.insert(name.content.clone(), (TypesKind::Function {ret: Box::from(return_type.kind.clone()), 
            params: params.iter().map(|x| {x.1.kind.clone()}).collect() }, unsafe { PointerValue::new(function_value.as_value_ref()) }));
        
        let mut inside_loop = false;
        let mut returns;
        
        if let Some(body) = body {
            function.new_block("entry", true);
            // Function params
            for (i, param) in params.iter().enumerate() {
                let value = function.function_value.get_nth_param(i as u32).unwrap();
                let ptr = self.builder.build_alloca(value.get_type(), "").expect("Failed to build alloca");
                self.builder.build_store(ptr, value).expect("Failed to store param");
                function.function_scope.definitions.insert(param.0.content.clone(), (param.1.kind.clone(), ptr));
            }
            
            let length = body.len();
            let mut enumer = body.into_iter().enumerate();
            let mut current = enumer.next().unwrap();

            while current.0 <= length {
                (inside_loop, returns) = self.visit_statement(&mut function, current.1, global_scope, None)?;
                if returns && (current.0 + 1 < length) {
                    self.warning(CodeWarning::dead_code(Self::ast_to_codepos(&enumer.next().unwrap().1), None));
                    break
                }
                if let Some(c) = enumer.next() { current = c; } else {break}
            }
        }
        
        Ok(())
    }
    
    fn warning(&self, code_warning: CodeWarning) {
        print_code_warn(code_warning, self.file_manager);
    }

    pub fn comp_ast<'a>(&'a self, module: Module<'a>, ast: Vec<AST>) -> CodeResult<Module<'a>> {
        let mut global_scope = Namespace::new();
        for branch in ast {
            match branch {
                AST::FunctionDef { ret, fmode, name, params, body } => {
                    self.visit_function_def(&module, name, fmode, ret, params, &mut global_scope, body)?;
                }
                _ => unreachable!()
            }
        }
        Ok(module)
    }
}
