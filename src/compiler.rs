use std::collections::HashMap;
use std::ops::Deref;
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder};
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{AsTypeRef, BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FloatType, FunctionType, IntType, StructType};
use inkwell::values::{AsValueRef, BasicValue, FloatValue, FunctionValue, InstructionOpcode, IntMathValue, IntValue, PointerValue};
use crate::codeviz::print_code_warn;
use crate::comp_errors::{CodeError, CodeResult, CodeWarning};
use crate::DevDebugLevel;
use crate::directives::{visit_directive, CompilationConfig};
use crate::filemanager::FileManager;
use crate::lexer::{tokenize, CodePosition, Token};
use crate::parser::{BinaryOp, Expression, ExpressionKind, FunctionMode, ModuleAccessVariant, Parser, Types, TypesKind, AST};

fn is_type_signed(typ: &TypesKind) -> bool {
    !matches!(typ, TypesKind::U32 | TypesKind::U64 | TypesKind::U8 | TypesKind::Bool)
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
    ret: TypesKind,
    function_scope: Namespace<'ctx>,
    name: String,
}

impl<'ctx> Function<'ctx> {
    pub fn new(compiler: &'ctx Compiler<'ctx>, function_value: FunctionValue<'ctx>, ret: TypesKind, name: String) -> Self {
        Self { compiler, function_value, ret, function_scope: Namespace::new(), name }
    }

    pub fn new_block(&self, block_name: &str, pae: bool) -> BasicBlock<'ctx> {
        let bb = self.compiler.context.append_basic_block(self.function_value, block_name);
        if pae { self.compiler.builder.position_at_end(bb) }
        bb
    }
}

#[derive(Debug)]
pub struct Namespace<'n> {
    pub definitions: HashMap<String, (TypesKind, PointerValue<'n>)>,
    pub structs: HashMap<String, StructType<'n>>,
    pub functions: HashMap<String, (FunctionValue<'n>, bool, TypesKind)>,
    pub struct_order: HashMap<String, Vec<(String, TypesKind)>>,
    pub modules: HashMap<String, Namespace<'n>>,
}

// Check local and global scope
fn check_ls_gs_defs<'a>(name: &str, ls: &'a Namespace<'a>, gs: &'a Namespace<'a>) -> Option<&'a (TypesKind, PointerValue<'a>)> {
    ls.definitions.get(name).or(gs.definitions.get(name))
}

impl<'ns> Namespace<'ns> {
    fn new() -> Self {
        Self { definitions: HashMap::new(), functions: HashMap::new(), struct_order: HashMap::new(), modules: HashMap::new(), structs: HashMap::new() }
    }
}

fn basic_value_box_into_int<'a>(value: Box<(dyn BasicValue<'a> + 'a)>) -> IntValue<'a> {
    value.as_basic_value_enum().into_int_value()
}

fn basic_value_box_into_float<'a>(value: Box<(dyn BasicValue<'a> + 'a)>) -> FloatValue<'a> {
    value.as_basic_value_enum().into_float_value()
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

    fn convert_type_normal(&self, typ: &TypesKind, global_scope: &Namespace, cpos: CodePosition) -> CodeResult<BasicTypeEnum> {
        // FUCK THIS SHIT
        unsafe {
            Ok(BasicTypeEnum::new(match typ {
                TypesKind::I32 | TypesKind::U32 => Ok(self.context.i32_type().as_type_ref()),
                TypesKind::F32 => Ok(self.context.f32_type().as_type_ref()),
                TypesKind::Void => Err(CodeError::void_type(cpos)),
                TypesKind::Struct { name } => {
                    let holder_scope = self.get_holder_scope(name, None, global_scope)?;
                    let x = holder_scope.structs.get(&name.last_name().unwrap()).map(|t| t.as_basic_type_enum());
                    Ok(x.unwrap().as_type_ref())
                }
                TypesKind::Function { ret, params } => {
                    Ok(self.convert_type_function(&ret.to_owned(), params.iter().map(|x| { x.to_owned() }).collect(), global_scope, cpos)?.ptr_type(AddressSpace::default()).as_type_ref())
                }
                TypesKind::Ptr(ptr) => {
                    Ok(self.convert_type_normal(ptr, global_scope, cpos)?.ptr_type(AddressSpace::default()).as_type_ref())
                }
                TypesKind::Pointer => Ok(self.context.ptr_type(AddressSpace::default()).as_type_ref()),
                TypesKind::I64 | TypesKind::U64 => Ok(self.context.i32_type().as_type_ref()),
                TypesKind::F64 => Ok(self.context.f64_type().as_type_ref()),
                TypesKind::U8 => Ok(self.context.i8_type().as_type_ref()),
                TypesKind::Bool => Ok(self.context.custom_width_int_type(1).as_type_ref()),
            }?))
        }
    }

    fn convert_type_function(&self, ret_type: &TypesKind, params: Vec<TypesKind>, global_scope: &Namespace, cpos: CodePosition) -> CodeResult<FunctionType> {
        let param_types: Vec<BasicMetadataTypeEnum> = params
            .iter()
            .map(|typ| {
                self.convert_type_normal(typ, global_scope, cpos).map(|x1| BasicMetadataTypeEnum::from(x1.as_basic_type_enum()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let t_ref = match ret_type {
            TypesKind::I32 | TypesKind::U32 => Ok(self.context.i32_type().fn_type(&param_types, false).as_type_ref()),
            TypesKind::F32 => Ok(self.context.f32_type().fn_type(&param_types, false).as_type_ref()),
            TypesKind::Void => Ok(self.context.void_type().fn_type(&param_types, false).as_type_ref()),
            TypesKind::Struct { name } => {
                let holder_scope = self.get_holder_scope(name, None, global_scope)?;
                let x = holder_scope.structs.get(&name.last_name().unwrap()).map(|t| Box::new(t.as_basic_type_enum()));
                Ok(x.unwrap().fn_type(&param_types, false).to_owned().as_type_ref())
            },
            TypesKind::Function { ret,params } => {
                Ok(self.convert_type_function(&ret.to_owned(), params.iter().map(|x| { x.to_owned() }).collect(), global_scope, cpos)?.ptr_type(AddressSpace::default()).fn_type(&param_types, false).as_type_ref())
            },
            TypesKind::Ptr(ptr) => {
                Ok(self.convert_type_normal(ptr, global_scope, cpos)?.ptr_type(AddressSpace::default()).fn_type(&param_types, false).as_type_ref())
            },
            TypesKind::Pointer => Ok(self.context.ptr_type(AddressSpace::default()).fn_type(&param_types, false).as_type_ref()),
            TypesKind::I64 | TypesKind::U64 => Ok(self.context.i64_type().fn_type(&param_types, false).as_type_ref()),
            TypesKind::F64 => Ok(self.context.f64_type().fn_type(&param_types, false).as_type_ref()),
            TypesKind::U8 => Ok(self.context.i8_type().fn_type(&param_types, false).as_type_ref()),
            TypesKind::Bool => Ok(self.context.custom_width_int_type(1).fn_type(&param_types, false).as_type_ref()),
        }?;
        unsafe { Ok(FunctionType::new(t_ref)) }
    }
    
    fn to_bool_int<'a>(&'a self, cond: IntValue<'a>) -> Box<dyn BasicValue + 'a> {
        Box::new(self.builder.build_int_compare(IntPredicate::NE, cond, self.context.custom_width_int_type(1).const_zero(), "is_nonzero").expect("Failed cond comp").as_basic_value_enum())
    }

    fn visit_bin_op<'a>(&'a self, code_position: CodePosition, lhs: Box<(dyn BasicValue<'a> + 'a)>, rhs: Box<(dyn BasicValue<'a> + 'a)>, op: BinaryOp, typ: TypesKind) -> CodeResult<(Box<dyn BasicValue + 'a>, TypesKind)> {
        let result: (Box<dyn BasicValue<'a>>, TypesKind) = match typ {
            TypesKind::I32 | TypesKind::U32 | TypesKind::I64 | TypesKind::U64 | TypesKind::U8 | TypesKind::Bool => {
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

            _ => return Err(CodeError::bin_op_on_non_primitive_type(code_position, typ)),
        };
        Ok(result)
    }
    
    fn null(&self) -> IntValue<'ctx> {
        self.context.i32_type().const_zero()
    }

    fn struct_access_type_mismatch_error(parent_cpos: &CodePosition, parent_type: &TypesKind) -> CodeError {
        let mut notes = vec!["Can only access elements from struct (pointers)".to_string()];
        if matches!(parent_type, TypesKind::Struct { .. }) {
            notes.push("As you are trying to access a field from a struct value, you might have forgotten to reference it?".to_string());
            CodeError::type_mismatch(parent_cpos, parent_type, &TypesKind::Ptr(Box::new(parent_type.clone())), notes)
        } else {
            todo!("Make type mismatch work");
            // CodeError::type_mismatch(parent_cpos, parent_type, &TypesKind::Struct { name: "struct".to_string() }, notes)
        }
    }

    fn resolve_mav_as_module<'a>(&self, mav: &ModuleAccessVariant, scope: &'a Namespace<'a>) -> CodeResult<&'a Namespace<'a>> {
        match mav {
            ModuleAccessVariant::Base { name, cpos } => {
                Ok(scope.modules.get(name).ok_or_else(|| { CodeError::prim_module_not_found(name, *cpos) })?)
            }
            ModuleAccessVariant::Double(a, b) => {
                let ns1 = self.resolve_mav_as_module(a, scope)?;
                let ns2 = self.resolve_mav_as_module(b, ns1)?;
                Ok(ns2)
            }
        }
    }

    fn resolve_mav<'a>(&self, mav: &ModuleAccessVariant, ls: Option<&'a Namespace<'a>>, scope: &'a Namespace<'a>) -> CodeResult<&'a (TypesKind, PointerValue<'a>)> {
        match mav {
            ModuleAccessVariant::Base { name, cpos } => {
                Ok(ls.and_then(|n1| {n1.definitions.get(name)}).or_else(|| { scope.definitions.get(name) })
                    .ok_or_else(|| { CodeError::prim_symbol_not_found(name, *cpos) })?)
            }
            ModuleAccessVariant::Double(a, b) => {
                let ns = self.resolve_mav_as_module(a, scope)?;
                Ok(self.resolve_mav(b, None, ns)?)
            }
        }
    }

    fn resolve_mav_func<'a>(&self, mav: &ModuleAccessVariant, ls: Option<&'a Namespace<'a>>, scope: &'a Namespace<'a>) -> CodeResult<&'a(FunctionValue<'a>, bool, TypesKind)> {
        match mav {
            ModuleAccessVariant::Base { name, cpos } => {
                Ok(ls.and_then(|n1| {n1.functions.get(name)}).or_else(|| { scope.functions.get(name) })
                    .ok_or_else(|| { CodeError::prim_symbol_not_found(name, *cpos) })?)
            }
            ModuleAccessVariant::Double(a, b) => {
                let ns = self.resolve_mav_as_module(a, scope)?;
                Ok(self.resolve_mav_func(b, None, ns)?)
            }
        }
    }

    fn get_holder_scope<'a>(&self, mav: &ModuleAccessVariant, ls: Option<&'a Namespace<'a>>, scope: &'a Namespace<'a>) -> CodeResult<&'a Namespace<'a>> {
        match mav {
            ModuleAccessVariant::Base { name, cpos } => {
                Ok(ls.and_then(|n1| {if n1.definitions.contains_key(name) || n1.structs.contains_key(name) {Some(n1)} else {None}}).or_else(|| {if scope.definitions.contains_key(name) || scope.structs.contains_key(name) {Some(scope)} else {None}})
                    .ok_or_else(|| { CodeError::prim_symbol_not_found(name, *cpos) })?)
            }
            ModuleAccessVariant::Double(a, _) => {
                let ns = self.resolve_mav_as_module(a, scope)?;
                Ok(ns)
            }
        }
    }

    fn visit_expr<'a>(&'a self, function: &'a Function<'a>, global_scope: &'a Namespace<'_>, expr: Expression, 
                      type_hint: Option<&TypesKind>, must_use: bool) -> CodeResult<(Box<dyn BasicValue + 'a>, TypesKind)> {
        Ok(match expr.expression {
            ExpressionKind::ModuleAccess(mav) => {
                let def = self.resolve_mav(&mav, Some(&function.function_scope), global_scope)?;
                let real_type = self.convert_type_normal(&def.0, global_scope, mav.ensured_compute_codeposition())?;
                (Box::new(self.builder.build_load(real_type.as_basic_type_enum(), def.1, "").expect("Failed to load")), def.0.clone())
            }
            ExpressionKind::IntNumber { value, .. } => {
                let (hint, vt) = hinted_int(type_hint, self.context);
                (Box::new(hint.const_int(value as u64, is_type_signed(&vt)).as_basic_value_enum()), vt)
            }
            ExpressionKind::FloatNumber { value, .. } => {
                let(hint, vt) = hinted_float(type_hint, self.context);
                (Box::new(hint.const_float(value).as_basic_value_enum()), vt)
            }
            ExpressionKind::BinaryOp { lhs, op, rhs } => {
                let cpos = rhs.code_position;
                let cpos_left = lhs.code_position;
                let (lhs_val, l_typ) = self.visit_expr(function, global_scope, *lhs, type_hint, true)?;
                let (rhs_val, r_typ) = self.visit_expr(function, global_scope, *rhs, Some(&l_typ), true)?;
                if l_typ != r_typ {
                    return Err(CodeError::type_mismatch(&cpos, &r_typ, &l_typ, vec!["This is because the right side must have the same type as the left side".to_string()]));
                }
                self.visit_bin_op(cpos_left.merge(cpos), lhs_val, rhs_val, op.1, l_typ)?
            }
            ExpressionKind::FunctionCall { name: mav, arguments } => {
                let def = self.resolve_mav_func(&mav, Some(&function.function_scope), global_scope)?;
                match &def.2 {
                    TypesKind::Function { params, ret } => {
                        if let Some(hint) = type_hint {
                            if must_use && *hint == TypesKind::Void {
                                return Err(CodeError::void_return(&expr.code_position));
                            }
                        }
                        let mut h_args = vec![];
                        if arguments.len() != params.len() {
                            return Err(CodeError::argument_count(mav.ensured_compute_codeposition(), arguments.len(), params.len()))
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
                        let call = self.builder.build_call(def.0, h_args.as_slice(), "").expect("Failed to load");
                        (Box::new(call.try_as_basic_value().left_or(self.null().as_basic_value_enum())), ret.as_ref().clone())
                    }
                    _ => Err(CodeError::symbol_not_a_function(&mav))?
                }
            }
            ExpressionKind::String(string) => {
                let global = self.builder.build_global_string_ptr(&string.content, "").expect("Failed to create global str ptr").as_pointer_value();
                (Box::new(global.as_basic_value_enum()), TypesKind::Ptr(Box::new(TypesKind::U8)))
            }
            ExpressionKind::CastExpr { expr, typ: new_type } => {
                let (value, old_type) = self.visit_expr(function, global_scope, *expr, None, true)?;
                let value = value.as_basic_value_enum();
                let real_new_typ = self.convert_type_normal(&new_type.kind, global_scope, new_type.cpos)?
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
                                    return Ok((Box::new(value), new_type.kind));
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
                                let target_type = self.convert_type_normal(ptr_type, global_scope, new_type.cpos)?
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
                    CodeError::invalid_cast(new_type.cpos, &new_type.kind, &old_type)
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

                let ptr_t = self.convert_type_normal(&def.0, global_scope, var.code_position)?;
                let real_deref_type = self.convert_type_normal(deref_type, global_scope, var.code_position)?;
                
                let ptr = self.builder.build_load(ptr_t.as_basic_type_enum(), def.1, "")
                    .expect("Load (deref) failed").as_basic_value_enum().into_pointer_value();
                let value = self.builder.build_load(real_deref_type.as_basic_type_enum(), ptr, "")
                    .expect("Load (deref) failed").as_basic_value_enum();
                (Box::new(value), *deref_type.clone())
            },
            ExpressionKind::New { name: mav, arguments } => {
                let cpos = mav.ensured_compute_codeposition();
                let mav_last_name = mav.last_name().clone().unwrap();
                let holder_scope = self.get_holder_scope(&mav, Some(&function.function_scope), global_scope)?;
                let virtual_struct_type = TypesKind::Struct { name: mav };
                let struct_type = self.convert_type_normal(&virtual_struct_type, global_scope, cpos)?
                    .as_basic_type_enum().into_struct_type();
                let expected = holder_scope.struct_order
                    .get(&mav_last_name)
                    .expect("Failed to get struct - which should not happen btw");

                let params = arguments.into_iter().enumerate()
                    .map(|(i, expr)| {
                        let cpos = expr.code_position;
                        let (e, t) = self.visit_expr(function, global_scope, expr, Some(&expected[i].1), true)?;
                        if expected[i].1 != t {
                            Err(CodeError::type_mismatch(&cpos, &t, &expected[i].1, vec![format!("The param nr. {i} must be of type `{}`", expected[i].1)]))
                        } else {
                            Ok((e, t))
                        }
                    })
                    .collect::<CodeResult<Vec<(Box<dyn BasicValue>, TypesKind)>>>()?;

                // Allocate space for the struct
                let struct_ptr = self.builder.build_alloca(struct_type, "struct_alloc").expect("Failed to struct alloc");

                for (i, (val, _ty)) in params.into_iter().enumerate() {
                    let gep = unsafe {
                        self.builder.build_struct_gep(struct_type, struct_ptr, i as u32, &format!("field_{i}"))
                            .expect("GEP failed - invalid field index?")
                    };
                    self.builder.build_store(gep, val.as_basic_value_enum()).expect("Failed store");
                }

                // Load the struct value (optional: you can return the pointer if that fits your ABI)
                let loaded_struct = self.builder.build_load(struct_type, struct_ptr, "load_struct").expect("Failed to load struct");

                (Box::new(loaded_struct), virtual_struct_type)
            }
            ExpressionKind::Access { parent, child, ptr } => {
                let parent_cpos = parent.code_position;
                let (parent_value, parent_type) = self.visit_expr(function, global_scope, *parent, None, true)?;
                let inner_struct_type;
                let (stct, stct_name) = match &parent_type {
                    TypesKind::Ptr(ptr) => {
                        match ptr.deref() {
                            TypesKind::Struct { name: mav } => {
                                inner_struct_type = ptr;
                                let holder_scope = self.get_holder_scope(mav, Some(&function.function_scope), global_scope)?;
                                (holder_scope.struct_order.get(&mav.last_name().unwrap()).unwrap(), mav)
                            }
                            _ => return Err(Self::struct_access_type_mismatch_error(&parent_cpos, &parent_type))
                        }
                    }
                    _ => return Err(Self::struct_access_type_mismatch_error(&parent_cpos, &parent_type))
                };
                if let Some(item) = stct.iter().position(|x1| {x1.0 == child.content}) {
                    let (_, field_type) = &stct[item];
                    let llvm_struct_type = self.convert_type_normal(inner_struct_type, global_scope, child.code_position)?.as_basic_type_enum();
                    let val_ptr = self.builder.build_struct_gep(llvm_struct_type, parent_value.as_basic_value_enum().into_pointer_value(), item as u32, "").expect("Failed to GEP struct item ptr");
                    let ret_typ = if ptr {TypesKind::Ptr(Box::new(field_type.clone()))} else { field_type.clone()};
                    if ptr {
                        (Box::new(val_ptr), ret_typ)
                    } else {
                        let llvm_field_type = self.convert_type_normal(field_type, global_scope, child.code_position)?.as_basic_type_enum();
                        let loaded = self.builder.build_load(llvm_field_type, val_ptr, "").expect("Failed to load struct access ptr");
                        (Box::new(loaded), ret_typ)
                    }
                } else {
                    return Err(CodeError::field_not_found(child, &stct_name.name()))
                }
            },
            ExpressionKind::Malloc { amount } => {
                let cpos = amount.code_position;
                let (value, typ) = self.visit_expr(function, global_scope, *amount, Some(&TypesKind::U32), false)?;
                if is_type_signed(&typ) {
                    return Err(CodeError::is_signed(cpos, typ))
                }
                let ptr = self.builder.build_array_malloc(self.context.i8_type(), value.as_basic_value_enum().into_int_value(), "").expect("Failed to build malloc");
                (Box::new(ptr.as_basic_value_enum()), TypesKind::Ptr(Box::new(TypesKind::U8)))
            }
            ExpressionKind::Free { var } => {
                // TODO: Maybe make this a statement?
                let cpos = var.code_position;
                let (value, typ) = self.visit_expr(function, global_scope, *var, Some(&TypesKind::Pointer), false)?;
                if !matches!(typ, TypesKind::Ptr(_) | TypesKind::Pointer) {
                    return Err(CodeError::can_only_free_pointers(cpos, typ))
                }
                self.builder.build_free(value.as_basic_value_enum().into_pointer_value()).expect("Failed to free");
                (value, typ)
            },
        })
    }

    fn visit_statement<'a>(&'ctx self, function: &mut Function<'ctx>, statement: AST, global_scope: &mut Namespace<'a>, after_block: Option<BasicBlock>, cur_block: Option<&BasicBlock>) -> CodeResult<(bool, bool)> where 'ctx: 'a {
        let mut inside_loop = after_block.is_some();
        let mut returns = false;
        // Returns: inside_loop, returns
        
        match statement {
            AST::VariableDef { name, value, typ } => {
                let mut typ_tok = name.code_position;

                let val = if let Some(value) = value {
                    let cpos = &value.code_position.clone();
                    let (expr, typ) = if let Some(typ) = typ {
                        typ_tok = typ.cpos;
                        let (expr, got_typ) = self.visit_expr(function, global_scope, value, Some(&typ.kind), true)?;
                        if got_typ != typ.kind { return Err(CodeError::type_mismatch(cpos, &got_typ, &typ.kind, vec![])) }
                        (expr, got_typ)
                    } else {
                        self.visit_expr(function, global_scope, value, None, true)?
                    };
                    let pointer = self.builder.build_alloca(self.convert_type_normal(&typ, global_scope, typ_tok)?.as_basic_type_enum(), "").expect("Can not allocate for var-define");
                    self.builder.build_store(pointer, expr.as_basic_value_enum()).expect("Failed to store value of variable define");
                    (typ, pointer)
                } else {
                    let typ = typ.unwrap();
                    (typ.kind.clone(), self.builder.build_alloca(self.convert_type_normal(&typ.kind, global_scope, typ_tok)?.as_basic_type_enum(), "").expect("Can not allocate for var-declare"))
                };
                function.function_scope.definitions.insert(name.content.clone(), val);
            }
            AST::VariableReassign { name, value } => {
                let holder_scope = self.get_holder_scope(&name, Some(&function.function_scope), global_scope)?;
                let cpos = &value.code_position.clone();
                let (typ, ptr) = holder_scope.definitions.get(&name.last_name().unwrap()).ok_or_else(|| {CodeError::prim_symbol_not_found(&name.name(), name.ensured_compute_codeposition())})?;
                let (data, got_type) = self.visit_expr(function, global_scope, value, Some(typ), true)?;
                if got_type != *typ { return Err(CodeError::type_mismatch(cpos, &got_type, typ, vec![format!("This is because `{}` was originally declared as {typ}", name.name())])) }
                self.builder.build_store(*ptr, data.as_basic_value_enum()).expect("Failed to store value of variable define");
            }
            AST::Expression { expr } => { 
                self.visit_expr(function, global_scope, expr, None, false)?;
            }
            AST::Return(value, ret_tok) => {
                if let Some(expr) = value {
                    let cpos = &expr.code_position.clone();
                    let (n_value, typ) = self.visit_expr(function, global_scope, expr, Some(&function.ret), true)?;
                    if typ != function.ret {
                        return Err(CodeError::type_mismatch(cpos, &typ, &function.ret, vec![format!("This is because the function `{}` has the return-type `{}`", function.name, function.ret)]))
                    }
                    self.builder.build_return(Some(n_value.deref())).expect("Failed to return");
                } else {
                    if function.ret != TypesKind::Void {
                        return Err(CodeError::non_void_ret(ret_tok, &function.name, &function.ret))
                    }
                    self.builder.build_return(None).expect("Failed to return NONE");
                }
                returns = true;
            },
            AST::CondLoop(cond) => {
                let cpos = cond.condition.code_position;
                inside_loop = true;
                
                let cond_block = function.new_block("cond", false);
                let body_block = function.new_block("body", false); // <- Added body block
                let after_block = function.new_block("after", false);

                self.builder.build_unconditional_branch(cond_block).expect("Failed to enter cond block");

                self.builder.position_at_end(cond_block);
                
                {
                    let (cond_value, typ) = self.visit_expr(function, global_scope, cond.condition, None, true)?;
                    if typ != TypesKind::Bool {
                        return Err(CodeError::conditions_must_be_bool(cpos, typ))
                    }
                    let condition = cond_value.as_basic_value_enum().into_int_value();
                    self.builder.build_conditional_branch(condition, body_block, after_block)
                        .expect("Failed to make conditional branch");
                }

                self.builder.position_at_end(body_block);
                for stmt in cond.body {
                    let (il, rets) = self.visit_statement(function, stmt, global_scope, Some(after_block), Some(&body_block))?;
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
                    
                    let (cond_val, vt) = self.visit_expr(
                        function,
                        global_scope,
                        cond_block.condition.clone(),
                        None,
                        true)?;
                    
                    if vt != TypesKind::Bool {
                        return Err(CodeError::conditions_must_be_bool(cond_block.condition.code_position, vt))
                    }
                    
                    let cond_val = cond_val.as_basic_value_enum().into_int_value();

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
                            Some(&body_bb)
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
                                Some(&else_bb)
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
                    return Err(CodeError::loop_stmt_outside_loop(&tok.code_position, &tok.token_type))
                }
            },
            AST::Continue(tok) => if let Some(current) = cur_block {
                self.builder.position_at_end(*current);
            } else {
                return Err(CodeError::loop_stmt_outside_loop(&tok.code_position, &tok.token_type))
            },
            _ => unreachable!(),
        }
        Ok((inside_loop, returns))
    }
    
    fn ast_to_codepos(ast: &AST) -> CodePosition {
        match ast {
            AST::Expression { expr } => expr.code_position,
            AST::FunctionDef { name, .. } => name.code_position,
            AST::VariableDef { name, .. } => name.code_position,
            AST::VariableReassign { name, .. } => name.ensured_compute_codeposition(),
            AST::Return(_, tok) => tok.code_position,
            AST::IfCondition { first, .. } => first.condition.code_position,
            AST::CondLoop(c) => c.condition.code_position,
            AST::Break(tok) => tok.code_position,
            AST::Continue(tok) => tok.code_position,
            AST::Struct { name, .. } => name.code_position,
            AST::Directive(d) => d.code_position,
            &AST::Import { module: m, .. } => m.code_position,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn visit_function_def<'a>(&'a self, module: &Module<'a>, name: &Token, fmode: FunctionMode, 
                                         return_type: Types, params: Vec<(&Token, Types)>, global_scope: &mut Namespace<'a>,
                                         body: Option<Vec<AST>>) -> CodeResult<()>
    {
        let fn_type = self.convert_type_function(&return_type.kind, params.iter().map(|x| {x.1.kind.clone()}).collect(), global_scope, name.code_position)?;
        
        if let Some((_, already_defined, _)) = global_scope.functions.get(&name.content) { 
            if *already_defined {
                return Err(CodeError::already_exists(true, name))
            }
        }
        
        let typ = TypesKind::Function {ret: Box::new(return_type.kind.clone()), params: params.iter().map(|x1| {x1.1.kind.clone()}).collect()};
        
        let function_value = module.add_function(&name.content, fn_type, Some(fmode.into()));
        let mut function = Function::new(self, function_value, return_type.kind.clone(), name.content.clone());

        global_scope.functions.insert(name.content.clone(), (function_value, body.is_some(), typ));
        // global_scope.definitions.insert(name.content.clone(), (TypesKind::Function {ret: Box::from(return_type.kind.clone()), 
        //     params: params.iter().map(|x| {x.1.kind.clone()}).collect() }, unsafe { PointerValue::new(function_value.as_value_ref()) }));
        
        let mut inside_loop = false;
        let mut returns = false;
        
        if let Some(body) = body {
            let entry = function.new_block("entry", true);
            // Function params
            for (i, param) in params.iter().enumerate() {
                let value = function.function_value.get_nth_param(i as u32).unwrap();
                let ptr = self.builder.build_alloca(value.get_type(), "").expect("Failed to build alloca");
                self.builder.build_store(ptr, value).expect("Failed to store param");
                function.function_scope.definitions.insert(param.0.content.clone(), (param.1.kind.clone(), ptr));
            }
            
            let length = body.len();
            if length != 0 {
                let mut enumer = body.into_iter().enumerate();
                let mut current = enumer.next().unwrap();

                while current.0 <= length {
                    (inside_loop, returns) = self.visit_statement(&mut function, current.1, global_scope, None, Some(&entry))?;
                    if returns && (current.0 + 1 < length) {
                        self.warning(CodeWarning::dead_code(Self::ast_to_codepos(&enumer.next().unwrap().1), None));
                        break;
                    }
                    if let Some(c) = enumer.next() { current = c; } else { break; }
                }

                if !returns && (return_type.kind != TypesKind::Void) {
                    return Err(CodeError::non_void_no_ret_func(name, &return_type.kind));
                } else if !returns && (return_type.kind == TypesKind::Void) {
                    self.builder.build_return(None).expect("Failed to empty attach return");
                }
            }
        }

        Ok(())
    }
    
    fn visit_struct<'a>(&self, name: &'a Token, members: Vec<(&'a Token, Types)>, global_scope: &'a mut Namespace<'ctx>) -> CodeResult<()> {
        let real_membs = members.iter()
            .map(|x| {
                self.convert_type_normal(&x.1.kind, global_scope, x.1.cpos)
                                     .map(|x1| x1.as_basic_type_enum())})
                    .collect::<CodeResult<Vec<BasicTypeEnum>>>()?;
        let asoc = members.iter().map(|x2| {(x2.0.content.clone(), x2.1.kind.clone())}).collect();
        global_scope.struct_order.insert(name.content.to_owned(), asoc);
        let typ = self.context.opaque_struct_type(&name.content);
        typ.set_body(real_membs.as_slice(), false);
        global_scope.structs.insert(name.content.clone(), typ);
        Ok(())
    }
    
    fn warning(&self, code_warning: CodeWarning) {
        print_code_warn(code_warning, self.file_manager);
    }

    pub fn comp_ast<'a>(&'a self, module: Module<'a>, ast: Vec<AST>, compilation_config: &mut CompilationConfig, file_manager: &FileManager, used_modules: &mut std::vec::Vec<inkwell::module::Module<'a>>) -> CodeResult<(Module<'a>, Namespace)> {
        let mut global_scope = Namespace::new();
        let mut should_do = true;
        
        for branch in ast {
            match branch {
                AST::FunctionDef { ret, fmode, name, params, body } if should_do => {
                    self.visit_function_def(&module, name, fmode, ret, params, &mut global_scope, body)?;
                }
                AST::Struct { name, members } if should_do => {
                    self.visit_struct(name, members, &mut global_scope)?;
                }
                AST::Import { module: m, path } if should_do => {
                    let path = if let Some(p) = path {
                        p.content.clone()
                    } else {format!("{m}.sl")};

                    let dev_debug_level = DevDebugLevel::Full;

                    let new_file_manager = FileManager::new_from(path).unwrap();

                    let tokens = tokenize(new_file_manager.get_content())?;

                    if dev_debug_level as u32 >= 2 {
                        println!("Parsed Tokens:\n{:#?}", tokens);
                    }

                    let parser = Parser::new(tokens, &new_file_manager);
                    let ast = parser.parse(&mut 0)?;

                    if dev_debug_level as u32 >= 2 {
                        println!("Parsed AST:\n{:#?}", ast);
                    }

                    let new_module = self.context.create_module(&m.content);
                    let (md, scope) = self.comp_ast(new_module, ast, compilation_config, &new_file_manager, used_modules).expect("FAILED");
                    
                    // Copy over DECLARED, but not defined functions. These do NOT get linked in.
                    let mut copied_functions = HashMap::new();
                    for func in &scope.functions {
                        if !func.1.1 {
                            let fval = md.get_function(func.0).unwrap();
                            let fval = module.add_function(func.0, fval.get_type(), Some(fval.get_linkage()));
                            // Use newly created function value, because the other one is freed
                            copied_functions.insert(func.0.clone(), (fval, func.1.1, func.1.2.clone()));
                        }
                    }
                    
                    // Link with other, which copies all the stuff (except declarations)
                    module.link_in_module(md).expect("Failed to link modules");
                    
                    // Copy over all elements from the scope
                    let mut copied_structs = HashMap::new();
                    for stct in scope.structs {
                        let styp = module.get_struct_type(&stct.0).unwrap();
                        copied_structs.insert(stct.0, styp);
                    }

                    let copied_struct_order = scope.struct_order;
                    let copied_modules = scope.modules;

                    for func in scope.functions {
                        if copied_functions.contains_key(&func.0) {continue}
                        let ftyp = module.get_function(&func.0).unwrap();
                        copied_functions.insert(func.0, (ftyp, func.1.1, func.1.2));
                    }

                    let mut copied_definitions = HashMap::new();
                    for def in scope.definitions {
                        println!("{:#?}", def);
                        let gval = module.get_global(&def.0).map(|t| t.as_pointer_value());
                        copied_definitions.insert(def.0, (def.1.0, gval.unwrap()));
                    }
                    
                    let new_scope = Namespace {definitions: copied_definitions, functions: copied_functions, struct_order: copied_struct_order, structs: copied_structs, modules: copied_modules};
                    
                    global_scope.modules.insert(m.content.clone(), new_scope);
                }
                // TODO: Add an optional body to directives
                AST::Directive(directive) => {
                    should_do = visit_directive(directive, compilation_config, file_manager)?;
                }
                _ if should_do => unreachable!(),
                _ => {should_do = true}
            }
        }
        Ok((module, global_scope))
    }

    pub fn compile<'a>(&'a self, module: Module<'a>, ast: Vec<AST>, compilation_config: &mut CompilationConfig, file_manager: &FileManager) -> CodeResult<Module<'a>> {
        let mut used_modules = vec![];
        let (md, _) = self.comp_ast(module, ast, compilation_config, &file_manager, &mut used_modules)?;
        Ok(md)
    }
}

fn llvm_link2_modules(m1: &Module, m2: &Module) -> Result<bool, String> {
    use inkwell::llvm_sys::linker::LLVMLinkModules2;

    // let char_ptr: *mut libc::c_char = ptr::null_mut();
    
    let code = unsafe { LLVMLinkModules2(m1.as_mut_ptr(), m2.as_mut_ptr()) };

    // if code == 1 {
    //     debug_assert!(!char_ptr.is_null());
// 
    //     unsafe { Err(char_ptr.as_ref().unwrap().to_string()) }
    // } else {
    //     Ok(())
    // }
    Ok(code == 0)
}