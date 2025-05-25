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

                TokenType::Directive => {
                    self.advance(pointer);
                    let directive = self.parse_directive(pointer)?;
                    statements.push(AST::Directive(directive));
                }

                TokenType::Import => {
                    let import_stmt = self.parse_import(pointer)?;
                    statements.push(import_stmt);
                }

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

    fn parse_import(&self, pointer: &mut usize) -> CodeResult<AST> {
        self.consume(pointer, TokenType::Import, None)?;
        
        let module = self.consume(pointer, TokenType::Identifier, None)?;
        
        let path = if self.match_token(pointer, TokenType::As)? { 
            Some(self.consume(pointer, TokenType::String, None)?)
        } else { None };
        
        Ok(AST::Import { module, path })
    }

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

    fn parse_function_call(&self, pointer: &mut usize, name: ModuleAccessVariant) -> CodeResult<Expression> {
        let start_pos = self.previous(pointer).unwrap().code_position;
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

        let end_pos = self.previous(pointer).unwrap().code_position;
        Ok((ExpressionKind::FunctionCall { name, arguments }).into_expression(start_pos.merge(end_pos)))
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
                    self.advance(pointer);
                    let mav = self.parse_mav(pointer)?;
                    *pointer -= 1;
                    if self.match_next_token(pointer, TokenType::LParen)? {
                        Ok(AST::Expression { expr: self.parse_function_call(pointer, mav)? })
                    } else if self.match_next_token(pointer, TokenType::Equals)? {
                        self.advance(pointer);
                        Ok(AST::VariableReassign {name: mav, value: self.parse_expression(pointer)?})
                    } else {
                        let cpos = mav.ensured_compute_codeposition();
                        self.warning(CodeWarning::new_unnecessary_code(
                            cpos,
                            None,
                        ));
                        Ok(AST::Expression { expr: Expression { expression: ExpressionKind::ModuleAccess(mav), code_position: cpos } })
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
                        return Err(CodeError::invalid_vardef(name.code_position, typ.is_some()))
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
                TokenType::LParen | TokenType::Free | TokenType::Malloc | TokenType::Star | TokenType::Ref => Ok(AST::Expression {expr: self.parse_expression(pointer)? }),
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
    
    fn parse_mav(&self, pointer: &mut usize) -> CodeResult<ModuleAccessVariant> {
        let ident = self.previous(pointer).unwrap();
        let parent = ModuleAccessVariant::Base { name: ident.content.to_owned(), cpos: ident.code_position };
        if self.match_token(pointer, TokenType::ModuleAccess)? {
            self.advance(pointer);
            Ok(ModuleAccessVariant::Double(Box::new(parent), Box::new(self.parse_mav(pointer)?)))
        } else {
            Ok(parent)
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
                TokenType::GreaterEquals | TokenType::LesserEquals | TokenType::And | TokenType::Or | TokenType::NotEquals => {
                    let op = self.advance(pointer).unwrap();
                    let right = self.parse_primary(pointer)?;
                    let cpos = node.code_position.merge(right.code_position);
                    node = (ExpressionKind::BinaryOp { lhs: Box::new(node), op: (op, op.token_type.to_binop().unwrap()), rhs: Box::new(right) }).into_expression(cpos);
                }
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

    // Helper function for recursive calls
    fn parse_directive(&'a self, pointer: &mut usize) -> CodeResult<Directive<'a>> {
        self.consume(pointer, TokenType::LParen, None)?;
        let start_pos_idx_recursive = *pointer - 1; // Start from the LParen itself (or the token after #)
        let name = self.consume(pointer, TokenType::Identifier, Some("Expected a directive name in nested directive".to_string()))?;

        let mut arguments: Vec<DirectiveExpr<'a>> = Vec::new();

        loop {
            if self.match_token(pointer, TokenType::RParen)? {
                break;
            }

            let current_token = self.current(pointer)
                .ok_or_else(|| CodeError::missing_token_error(self.previous(pointer).unwrap()))?;

            match current_token.token_type {
                TokenType::LParen => {
                    let nested_directive = self.parse_directive(pointer)?;
                    arguments.push(DirectiveExpr::NestedDirective(Box::new(nested_directive)));
                },
                TokenType::Identifier => {
                    self.advance(pointer);
                    arguments.push(DirectiveExpr::Literal(DirectiveArgType::Identifier {
                        value: current_token.content.clone(),
                        token: current_token,
                    }));
                },
                TokenType::NumberInt => {
                    self.advance(pointer);
                    let value = current_token.content.parse::<i64>().unwrap();
                    arguments.push(DirectiveExpr::Literal(DirectiveArgType::IntNumber {
                        value,
                        token: current_token,
                    }));
                },
                TokenType::NumberFloat => {
                    self.advance(pointer);
                    let value = current_token.content.parse::<f64>().unwrap();
                    arguments.push(DirectiveExpr::Literal(DirectiveArgType::FloatNumber {
                        value,
                        token: current_token,
                    }));
                },
                TokenType::String => {
                    self.advance(pointer);
                    arguments.push(DirectiveExpr::Literal(DirectiveArgType::String {
                        value: current_token.content.clone(),
                        token: current_token,
                    }));
                },
                _ => {
                    return Err(CodeError::new_unexpected_token_error(
                        current_token,
                        TokenType::Identifier,
                        Some("Expected an identifier, number, string, or nested directive as an argument".to_string()),
                    ));
                }
            }
        }
        let end_pos = self.previous(pointer).unwrap().code_position;
        let start_pos_token = self.tokens.get(start_pos_idx_recursive).unwrap().code_position;
        let code_position = start_pos_token.merge(end_pos);

        Ok(Directive { name, arguments, code_position })
    }

    fn parse_primary(&self, pointer: &mut usize) -> CodeResult<Expression> {
        let token = self.advance(pointer)
            .ok_or_else(|| CodeError::missing_token_error(self.previous(pointer).unwrap()))?;
            match token.token_type {
                TokenType::Ref => {
                    Ok(Expression {expression: ExpressionKind::Reference {
                        var: self.consume(pointer, TokenType::Identifier, Some("`&` is a ref-token, which must be followed by a variable to reference".to_string()))? }
                        , code_position: token.code_position.merge(self.tokens[*pointer - 1].code_position) })
                },
                TokenType::Star => {
                    Ok(Expression {expression: ExpressionKind::Dereference {
                        var: self.consume(pointer, TokenType::Identifier, Some("`*` is a deref-token, which must be followed by a variable to a pointer".to_string()))? }
                        , code_position: token.code_position.merge(self.tokens[*pointer - 1].code_position) })
                }
                TokenType::Malloc => {
                    Ok(Expression {expression: ExpressionKind::Malloc {
                        amount: Box::new(self.parse_expression(pointer)?) }
                        , code_position: token.code_position.merge(self.tokens[*pointer].code_position) })
                }
                TokenType::Free => {
                    Ok(Expression {expression: ExpressionKind::Free {
                        var: Box::new(self.parse_expression(pointer)?) }
                        , code_position: token.code_position.merge(self.tokens[*pointer].code_position) })
                }
                TokenType::New => {
                    let start = token.code_position;
                    self.advance(pointer);
                    let name = self.parse_mav(pointer)?;
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
                    let mav_name = self.parse_mav(pointer)?;
                    if let Some(next) = self.peek(pointer) {
                        if next.token_type == TokenType::LParen {
                            return self.parse_function_call(pointer, mav_name);
                        }
                    }
                    let code_position = mav_name.ensured_compute_codeposition();
                    Ok(Expression { expression: ExpressionKind::ModuleAccess(mav_name), code_position })
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
            else if self.match_token(pointer, TokenType::Identifier)? { Ok(TypesKind::Struct {name: self.parse_mav(pointer)? }) }
            else {Err(CodeError::not_a_type_error(&self.tokens[*pointer]))})?;
        
        let start = self.previous(pointer).unwrap().code_position;

        loop {
            if self.match_token(pointer, TokenType::Star)? {
                kind = TypesKind::Ptr(Box::new(kind))
            } else {
                break
            }
        }
        let cpos = start.merge(self.previous(pointer).unwrap().code_position);
        Ok(Types {kind, cpos })
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
            FunctionMode::Extern => {Linkage::External}
            FunctionMode::Default => {Linkage::Internal}
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
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
    Struct {name: ModuleAccessVariant},
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

#[derive(Debug, Clone)]
pub struct Types {
    pub kind: TypesKind,
    pub cpos: CodePosition
}

#[derive(Debug, Clone)]
pub enum ModuleAccessVariant {
    Base {
        name: String,
        cpos: CodePosition
    },
    Double(Box<ModuleAccessVariant>, Box<ModuleAccessVariant>)
}

// Ignore cpos when comparing
impl PartialEq for ModuleAccessVariant {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                ModuleAccessVariant::Base { name: name1, .. },
                ModuleAccessVariant::Base { name: name2, .. },
            ) => name1 == name2,

            (
                ModuleAccessVariant::Double(left1, right1),
                ModuleAccessVariant::Double(left2, right2),
            ) => left1 == left2 && right1 == right2,

            _ => false,
        }
    }
}

impl ModuleAccessVariant {
    fn spec_compute_codeposition(&self) -> Option<CodePosition> {
        match self {
            ModuleAccessVariant::Base { cpos, .. } => Some(cpos.clone()),
            ModuleAccessVariant::Double(left, right) => {
                match (left.spec_compute_codeposition(), right.spec_compute_codeposition()) {
                    (Some(c1), Some(c2)) => Some(c1.merge(c2)),
                    (Some(c), None) | (None, Some(c)) => Some(c),
                    (None, None) => None,
                }
            }
        }
    }
    
    pub fn ensured_compute_codeposition(&self) -> CodePosition {
        self.spec_compute_codeposition().unwrap()
    }

    fn collect_names(&self, names: &mut Vec<String>) {
        match self {
            ModuleAccessVariant::Base { name, .. } => names.push(name.clone()),
            ModuleAccessVariant::Double(left, right) => {
                left.collect_names(names);
                right.collect_names(names);
            }
        }
    }

    pub fn last_name(&self) -> Option<String> {
        match self {
            ModuleAccessVariant::Base { name, .. } => Some(name.to_owned()),
            ModuleAccessVariant::Double(_, right) => right.last_name(),
        }
    }
    
    pub fn name(&self) -> String {
        let mut names = Vec::new();
        self.collect_names(&mut names);
        names.join("::")
    }
}

impl Display for ModuleAccessVariant {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut names = Vec::new();
        self.collect_names(&mut names);
        write!(f, "{}", names.join("::"))
    }
}

#[derive(Debug, Clone)]
pub enum ExpressionKind<'a> {
    IntNumber {
        value: i64,
        token: &'a Token,
    },
    FloatNumber {
        value: f64,
        token: &'a Token,
    },
    // Identifier(&'a Token),
    String(&'a Token),
    ModuleAccess(ModuleAccessVariant),
    BinaryOp { lhs: Box<Expression<'a>>, op: (&'a Token, BinaryOp), rhs: Box<Expression<'a>> },
    CastExpr { expr: Box<Expression<'a>>, typ: Types },
    FunctionCall { name: ModuleAccessVariant, arguments: Vec<Expression<'a>> },
    Reference { var: &'a Token },
    Dereference { var: &'a Token },
    Malloc { amount: Box<Expression<'a>> },
    Free { var: Box<Expression<'a>> },
    New { name: ModuleAccessVariant, arguments: Vec<Expression<'a>> },
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

#[derive(Debug, PartialEq, Clone)]
pub enum DirectiveArgType<'a> {
    Identifier {
        value: String,
        token: &'a Token,
    },
    IntNumber {
        value: i64,
        token: &'a Token,
    },
    FloatNumber {
        value: f64,
        token: &'a Token,
    },
    String {
        value: String,
        token: &'a Token,
    },
}

#[derive(Eq, PartialEq)]
#[derive(Clone)]
pub enum VirtualDirectiveArgType {
    Identifier,
    IntNumber,
    FloatNumber,
    String,
    Directive,
}

impl Display for VirtualDirectiveArgType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            VirtualDirectiveArgType::Directive => write!(f, "directive"),
            VirtualDirectiveArgType::Identifier => write!(f, "identifier"),
            VirtualDirectiveArgType::String => write!(f, "string"),
            VirtualDirectiveArgType::FloatNumber => write!(f, "float-number"),
            VirtualDirectiveArgType::IntNumber => write!(f, "int-number"),
        }
    }
}

pub fn format_virtual_type_sig(name: &str, sig: Vec<VirtualDirectiveArgType>) -> String {
    // TODO: Remove the empty space if sig is empty
    format!("({name} {})", sig.iter().map(|x| {format!("`{x}`")}).collect::<Vec<String>>().join(" "))
}

#[derive(Debug, PartialEq, Clone)]
pub enum DirectiveExpr<'a> {
    Literal(DirectiveArgType<'a>),
    NestedDirective(Box<Directive<'a>>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Directive<'a> {
    pub name: &'a Token,
    pub arguments: Vec<DirectiveExpr<'a>>,
    pub code_position: CodePosition,
}

#[derive(Debug, Clone)]
pub enum AST<'a> {
    Expression { expr: Expression<'a > },
    FunctionDef {
        name: & 'a Token,
        fmode: FunctionMode,
        ret: Types,
        params: Vec<(& 'a Token, Types)>,
        body: Option<Vec<AST<'a>>>,
    },
    VariableDef { name: &'a Token, value: Option<Expression<'a>>, typ: Option<Types> },
    VariableReassign { name: ModuleAccessVariant, value: Expression<'a> },
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
        members: Vec<(&'a Token, Types)>
    },
    Directive(Directive<'a>),
    Import { module: & 'a Token, path: Option< & 'a Token> },
}