use std::collections::HashMap;
use std::ops::Deref;
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, FloatType, FunctionType, IntType};
use inkwell::values::{AsValueRef, BasicValue, FloatValue, FunctionValue, InstructionOpcode, IntValue, PointerValue};
use crate::comp_errors::{CodeError, CodeResult};
use crate::lexer::Token;
use crate::parser::{BinaryOp, Expression, FunctionMode, Types, TypesKind, AST};

enum PrimitiveErrors {
    TypeVoidUnallowed
}

type PrimRes<T> = Result<T, PrimitiveErrors>;

fn resolve_prim_res<'a>(result: Result<FunctionType<'a>, PrimitiveErrors>, tok: &Token) -> CodeResult<FunctionType<'a>> {
    result.map_err(|_| {CodeError::void_type(tok)})
}

pub fn hinted_int<'a>(hint: Option<&TypesKind>, ctx: &'a Context) -> (IntType<'a>, bool, TypesKind) {
    match hint {
        Some(h) => {
            match h {
                _ => (ctx.i32_type(), true, TypesKind::I32),
            }
        } None => (ctx.i32_type(), true, TypesKind::I32)
    }
}

pub fn hinted_float<'a>(hint: Option<&TypesKind>, ctx: &'a Context) -> (FloatType<'a>, TypesKind) {
    match hint {
        Some(h) => {
            match h {
                _ => (ctx.f32_type(), TypesKind::F32),
            }
        } None => (ctx.f32_type(), TypesKind::F32)
    }
}

pub struct Function<'ctx> {
    compiler: &'ctx Compiler<'ctx>,
    function_value: FunctionValue<'ctx>,
    ret: &'ctx TypesKind,
    function_scope: Namespace<'ctx>
}

impl<'ctx> Function<'ctx> {
    pub fn new(compiler: &'ctx Compiler<'ctx>, function_value: FunctionValue<'ctx>, ret: &'ctx TypesKind) -> Self {
        Self { compiler, function_value, ret, function_scope: Namespace::new() }
    }

    pub fn new_block(&self, block_name: &str) -> BasicBlock<'ctx> {
        let bb = self.compiler.context.append_basic_block(self.function_value, block_name);
        self.compiler.builder.position_at_end(bb);
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

pub struct Compiler<'ctx> {
    context: &'ctx Context,
    builder: &'ctx Builder<'ctx>,
    module_name: String,
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(context: &'ctx Context, builder: &'ctx Builder<'ctx>, module_name: String) -> Self {
        Self { context, builder, module_name }
    }

    fn convert_type_normal<'a>(&'a self, typ: &TypesKind) -> PrimRes<Box<dyn BasicType + 'a>> {
        match typ {
            TypesKind::I32 => Ok(Box::new(self.context.i32_type())),
            TypesKind::F32 => Ok(Box::new(self.context.f32_type())),
            TypesKind::Void => Err(PrimitiveErrors::TypeVoidUnallowed),
            TypesKind::Struct { .. } => todo!("Add structs"),
            TypesKind::Function { ret, params } => {
                Ok(Box::new(self.convert_type_function(&*ret.to_owned(), params.iter().map(|x| {x.to_owned()}).collect())?.ptr_type(AddressSpace::default())))
            },
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
            TypesKind::I32 => Ok(self.context.i32_type().fn_type(&param_types, false)),
            TypesKind::F32 => Ok(self.context.f32_type().fn_type(&param_types, false)),
            TypesKind::Void => Ok(self.context.void_type().fn_type(&param_types, false)),
            TypesKind::Struct { .. } => todo!("Add structs"),
            TypesKind::Function { ret,params } => {
                Ok(*Box::new(self.convert_type_function(&*ret.to_owned(), params.iter().map(|x| { x.to_owned() }).collect())?.ptr_type(AddressSpace::default()).fn_type(&param_types, false)))
            },
        }
    }

    fn visit_bin_op<'a>(&'a self, lhs: Box<(dyn BasicValue<'a> + 'a)>, rhs: Box<(dyn BasicValue<'a> + 'a)>, op: BinaryOp, typ: &TypesKind) -> CodeResult<Box<dyn BasicValue + 'a>> {
        todo!("Make bin-ops work");
        /*
        match typ {
            TypesKind::I32 => {
                match op {
                    BinaryOp::Eq => Box::new(self.builder.build_int_compare(IntPredicate::EQ, basic_value_box_into_int(lhs), basic_value_box_into_int(rhs), "")
                        .expect("SI-EQ failed")),
                    BinaryOp::Neq => Box::new(self.builder.build_int_compare(IntPredicate::NE, basic_value_box_into_int(lhs), basic_value_box_into_int(rhs), "")
                        .expect("SI-NEQ failed")),
                    BinaryOp::Gt => Box::new(self.builder.build_int_compare(IntPredicate::SGT, basic_value_box_into_int(lhs), basic_value_box_into_int(rhs), "")
                        .expect("SI-GT failed")),
                    BinaryOp::Gte => Box::new(self.builder.build_int_compare(IntPredicate::SGE, basic_value_box_into_int(lhs), basic_value_box_into_int(rhs), "")
                        .expect("SI-GTE failed")),
                    BinaryOp::Lt => Box::new(self.builder.build_int_compare(IntPredicate::SLT, basic_value_box_into_int(lhs), basic_value_box_into_int(rhs), "")
                        .expect("SI-LT failed")),
                    BinaryOp::Lte => Box::new(self.builder.build_int_compare(IntPredicate::SLE, basic_value_box_into_int(lhs), basic_value_box_into_int(rhs), "")
                        .expect("SI-LTE failed")),
                    BinaryOp::Add => Box::new(self.builder.build_int_add(basic_value_box_into_int(lhs), basic_value_box_into_int(rhs), "")
                        .expect("SI-SUB failed")),
                    BinaryOp::Sub => Box::new(self.builder.build_int_sub(basic_value_box_into_int(lhs), basic_value_box_into_int(rhs), "")
                        .expect("SI-Addition failed")),
                    BinaryOp::Div => Box::new(self.builder.build_int_signed_div(basic_value_box_into_int(lhs), basic_value_box_into_int(rhs), "")
                        .expect("SI-Addition failed")),
                    BinaryOp::Mul => Box::new(self.builder.build_int_mul(basic_value_box_into_int(lhs), basic_value_box_into_int(rhs), "")
                        .expect("SI-MUL failed")),
                    BinaryOp::And => Box::new(self.builder.build_and(basic_value_box_into_int(lhs), basic_value_box_into_int(rhs), "")
                        .expect("SI-AND failed")),
                    BinaryOp::Or => Box::new(self.builder.build_or(basic_value_box_into_int(lhs), basic_value_box_into_int(rhs), "")
                        .expect("SI-OR failed")),
                }
            }
            TypesKind
            ::F32 => {
                match op {
                    BinaryOp::Eq => Box::new(self.builder.build_float_compare(FloatPredicate::OEQ, basic_value_box_into_float(lhs), basic_value_box_into_float(rhs), "")
                        .expect("OF-EQ failed")),
                    BinaryOp::Neq => Box::new(self.builder.build_float_compare(FloatPredicate::ONE, basic_value_box_into_float(lhs), basic_value_box_into_float(rhs), "")
                        .expect("OF-NEQ failed")),
                    BinaryOp::Gt => Box::new(self.builder.build_float_compare(FloatPredicate::OGT, basic_value_box_into_float(lhs), basic_value_box_into_float(rhs), "")
                        .expect("OF-GT failed")),
                    BinaryOp::Gte => Box::new(self.builder.build_float_compare(FloatPredicate::OGE, basic_value_box_into_float(lhs), basic_value_box_into_float(rhs), "")
                        .expect("OF-GTE failed")),
                    BinaryOp::Lt => Box::new(self.builder.build_float_compare(FloatPredicate::OLT, basic_value_box_into_float(lhs), basic_value_box_into_float(rhs), "")
                        .expect("OF-LT failed")),
                    BinaryOp::Lte => Box::new(self.builder.build_float_compare(FloatPredicate::OLE, basic_value_box_into_float(lhs), basic_value_box_into_float(rhs), "")
                        .expect("OF-LTE failed")),
                    BinaryOp::Add => Box::new(self.builder.build_float_add(basic_value_box_into_float(lhs), basic_value_box_into_float(rhs), "")
                        .expect("OF-SUB failed")),
                    BinaryOp::Sub => Box::new(self.builder.build_float_sub(basic_value_box_into_float(lhs), basic_value_box_into_float(rhs), "")
                        .expect("OF-Addition failed")),
                    BinaryOp::Div => Box::new(self.builder.build_float_div(basic_value_box_into_float(lhs), basic_value_box_into_float(rhs), "")
                        .expect("OF-Addition failed")),
                    BinaryOp::Mul => Box::new(self.builder.build_float_mul(basic_value_box_into_float(lhs), basic_value_box_into_float(rhs), "")
                        .expect("OF-MUL failed")),
                    BinaryOp::And => {
                        let left_i = self.builder.build_cast(InstructionOpcode::FPToSI, basic_value_box_into_float(lhs), self.context.i32_type(), "").expect("Failed to cast rhs float32 to si32");
                        let right_i = self.builder.build_cast(InstructionOpcode::FPToSI, basic_value_box_into_float(rhs), self.context.i32_type(), "").expect("Failed to cast rhs float32 to si32");
                        Box::new(self.builder.build_and(left_i.into_int_value(), right_i.into_int_value(), "")
                            .expect("OF-AND failed"))
                    }
                    BinaryOp::Or => {
                        let left_i = self.builder.build_cast(InstructionOpcode::FPToSI, basic_value_box_into_float(lhs), self.context.i32_type(), "").expect("Failed to cast rhs float32 to si32");
                        let right_i = self.builder.build_cast(InstructionOpcode::FPToSI, basic_value_box_into_float(rhs), self.context.i32_type(), "").expect("Failed to cast rhs float32 to si32");
                        Box::new(self.builder.build_or(left_i.into_int_value(), right_i.into_int_value(), "")
                            .expect("OF-OR failed"))
                    }
                }
            }
            // Types::Void => {}
            // Types::Struct { .. } => {}
            // Types::Function { .. } => {}
            _ => {panic!("Can not perform binary operations on this type!")}
        }*/
    }
    
    fn null(&self) -> IntValue<'ctx> {
        self.context.i32_type().const_zero()
    }

    fn visit_expr<'a>(&'a self, function: &'a Function<'a>, global_scope: &'a Namespace<'_>, expr: Expression, 
                      type_hint: Option<&TypesKind>, must_use: bool) -> CodeResult<(Box<dyn BasicValue + 'a>, TypesKind)> {
        Ok(match expr {
            Expression::Identifier(name) => {
                let def = check_ls_gs_defs(&name.content, &function.function_scope, global_scope).expect("Variable does not exist");
                let real_type = self.convert_type_normal(&def.0).map_err(|e| {CodeError::void_type(name)})?;
                (Box::new(self.builder.build_load(real_type.as_basic_type_enum(), def.1, &name.content).expect("Failed to load")), def.0.clone())
            }
            Expression::IntNumber { value, token } => {
                let (hint, sign, vt) = hinted_int(type_hint, self.context);
                (Box::new(hint.const_int(value, sign).as_basic_value_enum()), vt)
            }
            Expression::FloatNumber { value, token } => {
                let(hint, vt) = hinted_float(type_hint, self.context);
                (Box::new(hint.const_float(value).as_basic_value_enum()), vt)
            }
            Expression::BinaryOp { lhs, op, rhs } => {
                let (lhs, l_typ) = self.visit_expr(function, global_scope, *lhs, type_hint, true)?;
                let (rhs, r_typ) = self.visit_expr(function, global_scope, *rhs, Some(&l_typ), true)?;
                if l_typ != r_typ {
                    panic!("Can not do a bin-op with different types!")
                }
                (Box::new(self.visit_bin_op(lhs, rhs, op.1, &l_typ)?.as_basic_value_enum()), l_typ)
            }
            Expression::FunctionCall { name, arguments } => {
                let def = check_ls_gs_defs(&name.content, &function.function_scope, global_scope).expect("Variable does not exist");
                match &def.0 {
                    TypesKind::Function { params, ret } => {
                        if let Some(hint) = type_hint {
                            if must_use && *hint == TypesKind::Void {
                                panic!("Void type may not be used as a return value!! This creates an error because the value of this call must be used")
                            }
                        }
                        let mut h_args = vec![];
                        for (count, arg) in arguments.into_iter().enumerate() {
                            let (value, typ) = self.visit_expr(function, global_scope, arg, Some(&params[count]), true)?;
                            if typ != params[count] {panic!("The function signature expects a value of another type here")}
                            h_args.push(value.deref().as_basic_value_enum().into());
                        }
                        if h_args.len() != params.len() {
                            panic!("The function expects a different amount of arguments")
                        }
                        // Call function, if it does not return a value (Void function) return null
                        // In that case the return value will NOT be used, because of other Void checks
                        let call = self.builder.build_call(check_ls_gs_func(&name.content, &function.function_scope, global_scope)
                            .expect("Failed to get function - which should exist btw").0, h_args.as_slice(), "").expect("Failed to load");
                        (Box::new(call.try_as_basic_value().left_or(self.null().as_basic_value_enum())), ret.as_ref().clone())
                    }
                    _ => panic!("This must be a function")
                }
            },
            Expression::String(_) => {todo!("Implement")}
            Expression::Type { .. } => {todo!("Implement")}
            Expression::CastExpr { .. } => {todo!("Implement")}
        })
    }

    fn visit_statement(&'ctx self, function: &mut Function<'ctx>, statement: AST, global_scope: &mut Namespace) -> CodeResult<()> {
        match statement {
            AST::FunctionDef { .. } => panic!("Function definition is not allowed here (YET)"),
            AST::VariableSet { name, value, typ } => {
                // let typ = if let Some(expr) = value {
                //     let (value, inferred) = self.visit_expr(function, global_scope, expr, typ, true);
                //     if typ.unwrap().kind != inferred.kind {panic!("Returned value does not match variable signature")}
                //     typ.unwrap()
                // } else {typ.unwrap()};
                // let ptr = self.builder.build_alloca(self.convert_type_normal(&typ).as_basic_type_enum(), &name).expect("Failed to create pointer");
                // self.builder.build_store(ptr, value.as_basic_value_enum()).expect("Failed to store value at pointer");
                // function.function_scope.definitions.insert(name.content.clone(), (typ, ptr));
            }
            AST::Expression { expr, position } => { 
                self.visit_expr(function, global_scope, expr, None, false)?;
            }
            AST::Return(value) => {
                if let Some(expr) = value {
                    let (value, typ) = self.visit_expr(function, global_scope, expr, Some(&function.ret), true)?;
                    if typ != *function.ret {panic!("Returned value does not match function signature")}
                    self.builder.build_return(Some(value.deref())).expect("Failed to return");
                } else {
                    if *function.ret != TypesKind::Void {panic!("This function must return a value!")}
                    self.builder.build_return(None).expect("Failed to return NONE");
                }
            },
            // AST::VarReassign { name, value } => {
            //     if let Some(addr) = check_ls_gs_defs(&name, &function.function_scope, global_scope) {
            //         let (value, typ) = self.visit_expr(function, global_scope, value, Some(&addr.0), true);
            //         if addr.0 != typ { panic!("Variable has been reassigned with another type; you may create a new variable with a different type, but the same name tho") }
            //         self.builder.build_store(addr.1, value.as_basic_value_enum()).expect("Failed to store reassign");
            //     }
            // }
        }
        Ok(())
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
        let mut function = Function::new(self, function_value, &return_type.kind);

        global_scope.functions.insert(name.content.clone(), (function_value, body.is_some()));
        global_scope.definitions.insert(name.content.clone(), (TypesKind::Function {ret: Box::from(return_type.kind.clone()), 
            params: params.iter().map(|x| {x.1.kind.clone()}).collect() }, unsafe { PointerValue::new(function_value.as_value_ref()) }));

        if let Some(body) = body {
            function.new_block("entry");
            for stmt in body {
                self.visit_statement(&mut function, stmt, global_scope)?;
            }
        }
        
        Ok(())
    }

    pub fn comp_ast<'a>(&'a self, module: Module<'a>, ast: Vec<AST>) -> CodeResult<Module<'a>> {
        let mut global_scope = Namespace::new();
        for branch in ast {
            match branch {
                AST::FunctionDef { ret, fmode, name, params, body } => {
                    self.visit_function_def(&module, name, fmode, ret, params, &mut global_scope, Some(body))?;
                }
                _ => panic!("This is not a top-level statement!"),
            }
        }
        Ok(module)
    }
}
