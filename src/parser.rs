use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use inkwell::module::Linkage;
use crate::codeviz::print_code_warn;
use crate::comp_errors::{CodeError, CodeResult, CodeWarning};
use crate::filemanager::FileManager;
use crate::lexer::{CodePosition, Token, TokenType};

static STATEMENT_TOKENS: [TokenType; 13] = [
    TokenType::If,
    TokenType::Else,
    TokenType::Elif,
    TokenType::For,
    TokenType::While,
    TokenType::Let,
    TokenType::Return,
    TokenType::Identifier,
    TokenType::String,
    TokenType::NumberInt,
    TokenType::NumberFloat,
    TokenType::LParen,
    TokenType::RParen
];

enum EndSAR {
    Brace(CodePosition),
    Semicolon(CodePosition),
    Nothing
}

pub struct Parser<'a> {
    tokens: Vec<Token>,
    file_manager: &'a FileManager,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: Vec<Token>, file_manager: &'a FileManager) -> Self {
        Self {
            tokens,
            file_manager,
        }
    }

    fn peek(&self, pointer: &usize) -> Option<&Token> {
        self.tokens.get(*pointer)
    }

    fn advance(&self, pointer: &mut usize) -> Option<&Token> {
        let token = self.tokens.get(*pointer);
        if token.is_some() {
            *pointer += 1;
        }
        token
    }

    fn match_token(&self, pointer: &mut usize, token_type: TokenType) -> CodeResult<bool> {
        self.is_done_err(pointer)?;
        if let Some(token) = self.peek(pointer) {
            if token.token_type == token_type {
                self.advance(pointer);
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn multi_match_token(&self, pointer: &mut usize, token_types: Vec<TokenType>) -> CodeResult<bool> {
        self.is_done_err(pointer)?;
        if let Some(token) = self.peek(pointer) {
            if token_types.contains(&token.token_type) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn match_next_token(&self, pointer: &mut usize, token_type: TokenType) -> CodeResult<bool> {
        self.is_done_err(pointer)?;
        if let Some(token) = self.tokens.get(*pointer + 1) {
            if token.token_type == token_type {
                self.advance(pointer);
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn consume(
        &self,
        pointer: &mut usize,
        expected: TokenType,
        note: Option<String>,
    ) -> CodeResult<&Token> {
        self.is_done_err(pointer)?;
        if self.match_token(pointer, expected)? {
            Ok(self.previous(pointer).unwrap())
        } else {
            Err(CodeError::new_unexpected_token_error(
                self.current(pointer).or(self.previous(pointer)).unwrap(),
                expected,
                note,
            ))
        }
    }

    fn previous(&self, pointer: &usize) -> Option<&Token> {
        self.tokens.get(*pointer - 1)
    }

    fn current(&self, pointer: &usize) -> Option<&Token> {
        self.tokens.get(*pointer)
    }

    fn warning(&self, code_warning: CodeWarning) {
        print_code_warn(code_warning, self.file_manager)
    }

    fn codepos_from_space(&self, s: usize, e: &usize, sub_off: usize) -> CodePosition {
        let start = self.tokens.get(s).unwrap().code_position;
        let end = self.tokens.get(*e - sub_off).unwrap().code_position;
        CodePosition {
            idx_start: start.idx_start,
            idx_end: end.idx_end,
            line_start: start.line_start,
            line_end: end.line_end,
            line_idx_start: start.line_idx_start,
            line_idx_end: end.line_idx_end,
        }
    }

    pub fn parse(&self, pointer: &mut usize) -> CodeResult<Vec<AST>> {
        let mut statements = Vec::new();

        while let Some(token) = self.peek(pointer) {
            match token.token_type {
                TokenType::Define => {
                    self.advance(pointer);
                    let func = self.parse_function(pointer)?;
                    statements.push(func);
                }
                
                TokenType::Struct => {
                    self.advance(pointer);
                    let func = self.parse_struct(pointer)?;
                    statements.push(func);
                }

                // // Parse import statements
                // TokenType::Import => {
                //     let import_stmt = self.parse_import(pointer)?;
                //     statements.push(import_stmt);
                // }

                _ => {
                    return Err(CodeError::toplevel_statement(token.code_position));
                }
            }
        }

        Ok(statements)
    }
    
    fn parse_struct(&self, pointer: &mut usize) -> CodeResult<AST> {
        let name = self.consume(pointer, TokenType::Identifier, None)?;
        self.consume(pointer, TokenType::LBrace, None)?;

        let mut members = vec![];
        
        loop {
            let mem_name = self.consume(pointer, TokenType::Identifier, None)?;
            self.consume(pointer, TokenType::Colon, None)?;
            let mem_type = self.parse_type(pointer)?;
            members.push((mem_name, mem_type));

            if self.match_token(pointer, TokenType::RBrace)? { break }
            self.consume(pointer, TokenType::Comma, Some("Struct definitions must end with `}` or continue using `,`".to_string()))?;
        };
        
        Ok(AST::Struct {name, members})
    }

    // // Parse import statement (assuming a simple import structure)
    // fn parse_import(&self, pointer: &mut usize) -> CodeResult<AST> {
    //     // Consume 'import' keyword
    //     self.consume(pointer, TokenType::Import, None)?;
//
    //     // Expect an identifier for the import (e.g., module name)
    //     let module_name = self.consume(pointer, TokenType::Identifier, None)?;
//
    //     // Optionally, handle import paths or other structures here if needed
    //     Ok(AST::Import(module_name))
    // }

    pub fn parse_function(&self, pointer: &mut usize) -> CodeResult<AST> {
        let fmode = if self.match_token(pointer, TokenType::Export)? { FunctionMode::Export }
        else if self.match_token(pointer, TokenType::Private)? { FunctionMode::Private }
        else if self.match_token(pointer, TokenType::Extern)? { FunctionMode::Extern }
        else { FunctionMode::Default };

        if self.multi_match_token(pointer, vec![TokenType::Extern, TokenType::Export, TokenType::Private])? {
            return Err(CodeError::function_overloaded(self.previous(pointer).unwrap()))
        }

        let name = self.consume(pointer, TokenType::Identifier, None)?;

        self.consume(pointer, TokenType::LParen, None)?;

        let args = self.parse_arguments(pointer)?;

        self.consume(pointer, TokenType::RParen, None)?;

        self.consume(pointer, TokenType::Colon, None)?;
        let ret = self.parse_type(pointer)?;

        let body = if self.match_token(pointer, TokenType::SemiColon)? { None } else { Some(self.parse_block(pointer)?) };

        Ok(AST::FunctionDef {
            name,
            fmode,
            ret,
            params: args,
            body
        })
    }

    fn parse_block(&self, pointer: &mut usize) -> CodeResult<Vec<AST>> {
        self.consume(pointer, TokenType::LBrace, None)?;

        let mut statements = Vec::new();

        while let Some(token) = self.peek(pointer) {
            if token.token_type == TokenType::RBrace {
                break;
            }

            let stmt = self.parse_statement(pointer)?;
            statements.push(*Box::new(stmt));

            if !self.match_token(pointer, TokenType::SemiColon)? {
                break;
            }
        }

        if self.match_token(pointer, TokenType::RBrace)? {
            Ok(statements)
        } else {
            match self.endstatement_analyzer(pointer) {
                EndSAR::Brace(codepos) => {
                    Err(CodeError::missing_ends_error(codepos.inc(), TokenType::RBrace))
                }
                EndSAR::Semicolon(codepos) => {
                    Err(CodeError::missing_ends_error(codepos.inc(), TokenType::SemiColon))
                }
                EndSAR::Nothing => {
                    Ok(statements)
                }
            }
        }
    }

    fn parse_function_call<'b>(&'b self, pointer: &mut usize, name: &'b Token) -> CodeResult<Expression<'b>> {
        self.consume(pointer, TokenType::LParen, None)?;
        let mut arguments = Vec::new();

        if !self.check(pointer, TokenType::RParen)? {
            loop {
                let expr = self.parse_expression(pointer)?;
                arguments.push(expr);
                if self.match_token(pointer, TokenType::Comma)? {
                    continue;
                }
                break;
            }
        }

        // Consume the closing ')'
        self.consume(pointer, TokenType::RParen, Some("Expected `)` after arguments".to_string()))?;

        let end_pos = self.previous(pointer).unwrap().code_position;
        let start_pos = name.code_position;
        let pos = start_pos.merge(end_pos);
        Ok((ExpressionKind::FunctionCall { name, arguments }).into_expression(pos))
    }

    fn check(&self, pointer: &usize, typ: TokenType) -> CodeResult<bool> {
        if let Some(tok) = self.peek(pointer) {
            Ok(tok.token_type == typ)
        } else {
            Err(CodeError::missing_token_error(self.previous(pointer).unwrap()))
        }
    }

    fn parse_return(&self, pointer: &mut usize) -> CodeResult<AST> {
        self.consume(pointer, TokenType::Return, None)?;
        if self.multi_match_token(pointer, vec![TokenType::SemiColon, TokenType::RBrace])? {
            Ok(AST::Return(None, &self.tokens[*pointer - 1]))
        } else {
            Ok(AST::Return(Some(self.parse_expression(pointer)?), &self.tokens[*pointer - 1]))
        }
    }

    fn endstatement_analyzer(&self, pointer: &usize) -> EndSAR {
        /*
        Check if next token is a statement / Expression.
        If it is, a semicolon should be placed before it.
        If the file is ending, it should definitely be a brace!
        TODO: If the token is a global token, it should also be a brace
        */
        let r = self.peek(pointer);
        if r.is_none() {
            EndSAR::Brace(self.previous(pointer).unwrap().code_position)
        } else {
            let tok = &r.unwrap();
            if STATEMENT_TOKENS.contains(&tok.token_type) {
                EndSAR::Semicolon(self.previous(pointer).unwrap().code_position)
            } else {
                EndSAR::Nothing
            }
        }
    }

    fn parse_statement(&self, pointer: &mut usize) -> CodeResult<AST> {
        let token = self.peek(pointer);

        if let Some(token) = token {
            match token.token_type {
                TokenType::Identifier => {
                    if self.match_next_token(pointer, TokenType::LParen)? {
                        Ok(AST::Expression { expr: self.parse_function_call(pointer, token)? })
                    } else if self.match_next_token(pointer, TokenType::Equals)? {
                        self.advance(pointer);
                        Ok(AST::VariableReassign {name: token, value: self.parse_expression(pointer)?})
                    } else {
                        let a = *pointer;
                        let res = self.parse_expression(pointer);
                        let cpos = self.codepos_from_space(a, pointer, 1);
                        self.warning(CodeWarning::new_unnecessary_code(
                            cpos,
                            None,
                        ));
                        Ok(AST::Expression { expr: res? })
                    }
                }
                TokenType::NumberInt | TokenType::NumberFloat => {
                    let a = *pointer;
                    let res = self.parse_expression(pointer);
                    let cpos = self.codepos_from_space(a, pointer, 1);
                    self.warning(CodeWarning::new_unnecessary_code(
                        cpos,
                        None,
                    ));
                    Ok(AST::Expression { expr: res? })
                }
                TokenType::String => {
                    let a = *pointer;
                    let res = self.parse_expression(pointer);
                    let cpos = self.codepos_from_space(a, pointer, 1);
                    self.warning(CodeWarning::new_unnecessary_code(
                        cpos,
                        None,
                    ));
                    Ok(AST::Expression { expr: res? })
                }
                TokenType::Let => {
                    self.advance(pointer);
                    let name = self.consume(pointer, TokenType::Identifier, None)?;
                    let typ = if self.match_token(pointer, TokenType::Colon)? {
                        Some(self.parse_type(pointer)?)
                    } else {None};
                    let value = if self.match_token(pointer, TokenType::Equals)? {
                        Some(self.parse_expression(pointer)?)
                    } else {None};
                    if typ.is_none() && value.is_none() {
                        return Err(CodeError::invalid_vardef(name, typ.is_some()))
                    }
                    Ok(AST::VariableDef {name, typ, value})
                }
                TokenType::While => {
                    self.advance(pointer);
                    let condition = self.parse_expression(pointer)?;
                    let body = self.parse_block(pointer)?;
                    Ok(AST::CondLoop(CondBlock {condition, body}))
                }
                TokenType::Break => {
                    Ok(AST::Break(self.consume(pointer, TokenType::Break, None)?))
                }
                TokenType::Continue => {
                    Ok(AST::Continue(self.consume(pointer, TokenType::Continue, None)?))
                }
                TokenType::If => self.parse_if_case(pointer),
                TokenType::Return => self.parse_return(pointer),
                _o => Err(CodeError::new_unexpected_token_error(
                    token,
                    TokenType::Statement,
                    Some("Expected some sort of statement".to_string()),
                )),
            }
        } else {
            Err(CodeError::missing_token_error(
                self.previous(pointer).unwrap(),
            ))
        }
    }
    
    fn parse_if_case(&self, pointer: &mut usize) -> CodeResult<AST> {
        self.advance(pointer);
        
        let first = CondBlock {condition: self.parse_expression(pointer)?, body: self.parse_block(pointer)?};
        
        let mut elif = vec![];
        loop {
            if self.match_token(pointer, TokenType::Elif)? {
                elif.push(CondBlock {condition: self.parse_expression(pointer)?, body: self.parse_block(pointer)?})
            }
            else {break}
        };

        let other = if self.match_token(pointer, TokenType::Else)? {
            Some(self.parse_block(pointer)?)
        } else { None };
        
        Ok(AST::IfCondition {first, other, elif})
    }

    fn is_done(&self, pointer: &usize) -> bool {
        (*pointer - 1) == self.tokens.len()
    }

    fn is_done_err(&self, pointer: &usize) -> CodeResult<()> {
        if self.is_done(pointer) {
            Err(CodeError::missing_token_error(
                self.previous(pointer).unwrap(),
            ))
        } else {
            Ok(())
        }
    }

    fn parse_arguments(&self, pointer: &mut usize) -> CodeResult<Vec<(&Token, Types)>> {
        let mut arguments = Vec::new();

        while let Some(token) = self.peek(pointer) {
            if token.token_type == TokenType::RParen {
                break;
            }

            let name = self.consume(pointer, TokenType::Identifier, None)?;
            self.consume(pointer, TokenType::Colon, None)?;
            let arg_type = self.parse_type(pointer)?;

            arguments.push((name, arg_type));

            if !self.match_token(pointer, TokenType::Comma)? {
                break;
            }
        }

        Ok(arguments)
    }

    fn parse_expression(&self, pointer: &mut usize) -> CodeResult<Expression> {
        let a = *pointer;
        let term = self.parse_term(pointer)?;
        if self.match_token(pointer, TokenType::As)? {
            let cpos = self.codepos_from_space(a, pointer, 1);
            Ok((ExpressionKind::CastExpr {expr: Box::new(term), typ: self.parse_type(pointer)?}).into_expression(cpos))
        } else {
            Ok(term)
        }
    }

    fn parse_term(&self, pointer: &mut usize) -> CodeResult<Expression> {
        let mut node = self.parse_factor(pointer)?;

        while let Some(token) = self.peek(pointer) {
            match token.token_type {
                TokenType::Plus | TokenType::Minus => {
                    let op = self.advance(pointer).unwrap();
                    let right = self.parse_factor(pointer)?;
                    let cpos = node.code_position.merge(right.code_position);
                    node = (ExpressionKind::BinaryOp { lhs: Box::new(node), op: (op, op.token_type.to_binop().unwrap()), rhs: Box::new(right) }).into_expression(cpos);
                }
                _ => break,
            }
        }
        Ok(node)
    }

    fn parse_factor(&self, pointer: &mut usize) -> CodeResult<Expression> {
        let mut node = self.parse_primary(pointer)?;

        while let Some(token) = self.peek(pointer) {
            match token.token_type {
                TokenType::Star | TokenType::Slash | TokenType::DoubleEquals | TokenType::Lesser | TokenType::Greater |
                TokenType::GreaterEquals | TokenType::LesserEquals | TokenType::And | TokenType::Or  => {
                    let op = self.advance(pointer).unwrap();
                    let right = self.parse_primary(pointer)?;
                    let cpos = node.code_position.merge(right.code_position);
                    node = (ExpressionKind::BinaryOp { lhs: Box::new(node), op: (op, op.token_type.to_binop().unwrap()), rhs: Box::new(right) }).into_expression(cpos);
                }
                /*
                else if next.token_type == TokenType::Dot {
                    self.consume(pointer, TokenType::Dot, None)?;
                    let child = self.consume(pointer, TokenType::Identifier, Some("Can only access identifiers".to_string()))?;
                    return Ok(Expression { expression : ExpressionKind::Access {parent: name_tok, child, ptr: false}, code_position: name_tok.code_position.merge(child.code_position)})
                } else if next.token_type == TokenType::Relative {
                    self.consume(pointer, TokenType::Relative, None)?;
                    let child = self.consume(pointer, TokenType::Identifier, Some("Can only access identifiers".to_string()))?;
                    return Ok(Expression { expression : ExpressionKind::Access {parent: name_tok, child, ptr: true}, code_position: name_tok.code_position.merge(child.code_position)})
                }
                 */
                _ => break,
            }
        }
        let cpos = node.code_position;
        if self.match_token(pointer, TokenType::Dot)? {
            let child = self.consume(pointer, TokenType::Identifier, Some("Can only access identifiers".to_string()))?;
            return Ok(Expression { expression : ExpressionKind::Access {parent: Box::new(node), child, ptr: false}, code_position: cpos.merge(child.code_position)})
        } else if self.match_token(pointer, TokenType::Relative)? {
            let child = self.consume(pointer, TokenType::Identifier, Some("Can only access identifiers".to_string()))?;
            return Ok(Expression { expression : ExpressionKind::Access {parent: Box::new(node), child, ptr: true}, code_position: cpos.merge(child.code_position)})
        }
        Ok(node)
    }

    fn parse_primary(&self, pointer: &mut usize) -> CodeResult<Expression> {
        let token = self.advance(pointer)
            .ok_or_else(|| CodeError::missing_token_error(self.previous(pointer).unwrap()))?;
            match token.token_type {
                TokenType::Ref => {
                    Ok(Expression {expression: ExpressionKind::Reference { 
                        var: self.consume(pointer, TokenType::Identifier, Some("`&` is a ref-token, which must be followed by a variable to reference".to_string()))? }
                        , code_position: token.code_position.merge(self.tokens[*pointer].code_position) })
                },
                TokenType::Star => {
                    Ok(Expression {expression: ExpressionKind::Dereference {
                        var: self.consume(pointer, TokenType::Identifier, Some("`*` is a deref-token, which must be followed by a variable to a pointer".to_string()))? }
                        , code_position: token.code_position.merge(self.tokens[*pointer].code_position) })
                }
                TokenType::New => {
                    let start = token.code_position;
                    let name = self.consume(pointer, TokenType::Identifier, None)?;
                    self.consume(pointer, TokenType::LParen, None)?;
                    let mut arguments = Vec::new();
                    if !self.check(pointer, TokenType::RParen)? {
                        loop {
                            let expr = self.parse_expression(pointer)?;
                            arguments.push(expr);
                            if self.match_token(pointer, TokenType::Comma)? {
                                continue;
                            }
                            break;
                        }
                    }
                    self.consume(pointer, TokenType::RParen, Some("Expected `)` after arguments".to_string()))?;

                    Ok((ExpressionKind::New {name, arguments}).into_expression(start.merge(self.tokens[*pointer - 1].code_position)))
                }
                TokenType::Identifier => {
                    let name_tok = token;
                    if let Some(next) = self.peek(pointer) {
                        if next.token_type == TokenType::LParen {
                            return self.parse_function_call(pointer, name_tok);
                        }
                    }
                    Ok(Expression { expression: ExpressionKind::Identifier(name_tok), code_position: name_tok.code_position })
                }
                TokenType::NumberInt => {
                    let val = token.content.parse().unwrap();
                    Ok((ExpressionKind::IntNumber { value: val, token }).into_expression(token.code_position))
                }
                TokenType::NumberFloat => {
                    let val = token.content.parse().unwrap();
                    Ok((ExpressionKind::FloatNumber { value: val, token }).into_expression(token.code_position))
                }
                TokenType::LParen => {
                    let expr = self.parse_expression(pointer)?;
                    self.consume(pointer, TokenType::RParen, None)?;
                    Ok(expr)
                }
                TokenType::String => Ok(Expression {expression: ExpressionKind::String(token), code_position: token.code_position}),
                _ => Err(CodeError::new_unexpected_token_error(
                    self.previous(pointer).unwrap(),
                    TokenType::Expression,
                    Some(
                        "You may add a literal (number), string, variable, or a term here"
                            .to_string(),
                    ),
                )),
            }
        }

    fn parse_type(&self, pointer: &mut usize) -> CodeResult<Types> {
        let mut kind =
            (if self.match_token(pointer, TokenType::I32)? { Ok(TypesKind::I32) }
            else if self.match_token(pointer, TokenType::F32)? { Ok(TypesKind::F32) }
            else if self.match_token(pointer, TokenType::F64)? { Ok(TypesKind::F64) }
            else if self.match_token(pointer, TokenType::I64)? { Ok(TypesKind::I64) }
            else if self.match_token(pointer, TokenType::U8)? { Ok(TypesKind::U8) }
            else if self.match_token(pointer, TokenType::U32)? { Ok(TypesKind::U32) }
            else if self.match_token(pointer, TokenType::U64)? { Ok(TypesKind::U64) }
            else if self.match_token(pointer, TokenType::Void)? { Ok(TypesKind::Void) }
            else if self.match_token(pointer, TokenType::Ptr)? { Ok(TypesKind::Pointer) }
            else if self.match_token(pointer, TokenType::Bool)? { Ok(TypesKind::Bool) }
            else if self.match_token(pointer, TokenType::Identifier)? { Ok(TypesKind::Struct {name: self.tokens[*pointer - 1].content.clone() }) }
            else {Err(CodeError::not_a_type_error(&self.tokens[*pointer]))})?;

        loop {
            if self.match_token(pointer, TokenType::Star)? {
                kind = TypesKind::Ptr(Box::new(kind))
            } else {
                break
            }
        }

        Ok(Types::new(kind, &self.tokens[*pointer - 1]))
    }
}

#[derive(Debug, Clone)]
#[derive(Hash)]
pub enum BinaryOp {
    Eq,
    Neq,
    Gt,
    Lt,
    Gte,
    Lte,
    Add,
    Sub,
    Div,
    Mul,
    And,
    Or
}

impl TokenType {
    fn to_binop(&self) -> Option<BinaryOp> {
        match self {
            TokenType::Star => Some(BinaryOp::Mul),
            TokenType::Minus => Some(BinaryOp::Sub),
            TokenType::Plus => Some(BinaryOp::Add),
            TokenType::Slash => Some(BinaryOp::Div),
            TokenType::And => Some(BinaryOp::And),
            TokenType::Or => Some(BinaryOp::Or),
            // TokenType::Exclamation => Some(BinaryOp::Not),
            TokenType::DoubleEquals => Some(BinaryOp::Eq),
            TokenType::NotEquals => Some(BinaryOp::Neq),
            TokenType::Greater => Some(BinaryOp::Gt),
            TokenType::Lesser => Some(BinaryOp::Lt),
            TokenType::GreaterEquals => Some(BinaryOp::Gte),
            TokenType::LesserEquals => Some(BinaryOp::Lte),
            _ => None
        }
    }
}

#[derive(Debug, Hash, Copy, Clone)]
pub enum FunctionMode {
    Private,
    Export,
    Extern,
    Default,
}

impl Into<Linkage> for FunctionMode {
    fn into(self) -> Linkage {
        match self {
            FunctionMode::Private => {Linkage::Private}
            FunctionMode::Export => {Linkage::External}
            FunctionMode::Extern => {Linkage::AvailableExternally}
            FunctionMode::Default => {Linkage::External}
        }
    }
}

#[derive(Debug, Hash, Clone)]
#[derive(PartialEq)]
pub enum TypesKind {
    I32,
    F32,
    U32,
    I64,
    F64,
    U64,
    U8,
    Void,
    Ptr(Box<TypesKind>),
    Pointer,
    Struct {name: String},
    Function {ret: Box<TypesKind>, params: Vec<TypesKind>},
    Bool,
}

impl Display for TypesKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            TypesKind::I32 => write!(f, "i32"),
            TypesKind::F32 => write!(f, "f32"),
            TypesKind::F64 => write!(f, "f64"),
            TypesKind::I64 => write!(f, "i64"),
            TypesKind::U32 => write!(f, "u32"),
            TypesKind::U64 => write!(f, "u64"),
            TypesKind::U8 => write!(f, "u8"),
            TypesKind::Void => write!(f, "void"),
            TypesKind::Struct { name } => write!(f, "{}", name),
            TypesKind::Function { .. } => write!(f, "function"),
            TypesKind::Ptr(ptr) => write!(f, "{}*", ptr),
            TypesKind::Pointer => write!(f, "ptr"),
            TypesKind::Bool => write!(f, "bool"),
        }
    }
}

#[derive(Debug, Hash, Clone)]
pub struct Types<'a> {
    pub kind: TypesKind,
    pub token: &'a Token
}

impl<'a> Types<'a> {
    pub fn new(kind: TypesKind, token: &'a Token) -> Self {
        Self {kind, token}
    }
}

#[derive(Debug, Clone)]
pub enum ExpressionKind<'a> {
    IntNumber {
        value: u64,
        token: &'a Token,
    },
    FloatNumber {
        value: f64,
        token: &'a Token,
    },
    Identifier(&'a Token),
    String(&'a Token),
    Type { typ: Types<'a>, token: &'a Token },
    BinaryOp { lhs: Box<Expression<'a>>, op: (&'a Token, BinaryOp), rhs: Box<Expression<'a>> },
    CastExpr { expr: Box<Expression<'a>>, typ: Types<'a> },
    FunctionCall { name: & 'a Token, arguments: Vec<Expression<'a>> },
    Reference { var: &'a Token },
    Dereference { var: &'a Token },
    New { name: & 'a Token, arguments: Vec<Expression<'a>> },
    Access { parent: Box<Expression<'a>>, child: &'a Token, ptr: bool }
}

impl<'a> ExpressionKind<'a> {
    pub fn into_expression(self, cpos: CodePosition) -> Expression<'a> {
        Expression {expression: self, code_position: cpos}
    }
}

#[derive(Debug, Clone)]
pub struct Expression<'a> {
    pub expression: ExpressionKind<'a>,
    pub code_position: CodePosition
}

#[derive(Debug, Clone)]
pub struct CondBlock<'a> {
    pub condition: Expression<'a>,
    pub body: Vec<AST<'a>>
}

#[derive(Debug, Clone)]
pub enum AST<'a> {
    Expression { expr: Expression<'a > },
    FunctionDef {
        name: & 'a Token,
        fmode: FunctionMode,
        ret: Types<'a>,
        params: Vec<(& 'a Token, Types<'a>)>,
        body: Option<Vec<AST<'a>>>,
    },
    VariableDef { name: & 'a Token, value: Option<Expression<'a>>, typ: Option<Types<'a>> },
    VariableReassign { name: & 'a Token, value: Expression<'a> },
    Return(Option<Expression<'a>>, &'a Token),
    IfCondition {
        first: CondBlock<'a>,
        other: Option<Vec<AST<'a>>>,
        elif: Vec<CondBlock<'a>>
    },
    CondLoop(CondBlock<'a>),
    Break(&'a Token),
    Continue(&'a Token),
    Struct {
        name: &'a Token,
        members: Vec<(&'a Token, Types<'a>)>
    },
}