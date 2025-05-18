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
                // Parse function definitions
                TokenType::Define => {
                    self.advance(pointer);
                    let func = self.parse_function(pointer)?;
                    statements.push(func);
                }

                // // Parse import statements
                // TokenType::Import => {
                //     let import_stmt = self.parse_import(pointer)?;
                //     statements.push(import_stmt);
                // }

                _ => {
                    return Err(CodeError::placeholder());
                }
            }
        }

        Ok(statements)
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

        let body = self.parse_block(pointer)?;

        Ok(AST::FunctionDef {
            name,
            fmode,
            ret,
            args,
            body,
        })
    }

    fn parse_block(&self, pointer: &mut usize) -> CodeResult<Vec<Box<AST>>> {
        self.consume(pointer, TokenType::LBrace, None)?;

        let mut statements = Vec::new();

        while let Some(token) = self.peek(pointer) {
            if token.token_type == TokenType::RBrace {
                break;
            }

            let stmt = self.parse_statement(pointer)?;
            statements.push(Box::new(stmt));

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

    fn parse_function_call(&self, pointer: &mut usize) -> CodeResult<Expression> {
        let name = self.previous(pointer).unwrap();
        self.consume(pointer, TokenType::LParen, None)?;
        let mut arguments = vec![];
        while let Some(_tok) = self.peek(pointer) {
            arguments.push(self.parse_expression(pointer)?);
            if self.match_token(pointer, TokenType::RParen)? {
                break;
            }
            self.consume(pointer, TokenType::Comma, Some("Add a comma".to_string()))?;
        }
        Ok(Expression::FunctionCall {name, arguments})
    }

    fn parse_return(&self, pointer: &mut usize) -> CodeResult<AST> {
        self.consume(pointer, TokenType::Return, None)?;
        if self.multi_match_token(pointer, vec![TokenType::SemiColon, TokenType::RBrace])? {
            Ok(AST::Return(None))
        } else {
            Ok(AST::Return(Some(self.parse_expression(pointer)?)))
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
                        let a = *pointer;
                        let expr = self.parse_function_call(pointer)?;
                        let position = self.codepos_from_space(a, pointer, 1);
                        Ok(AST::Expression { expr, position })
                    } else {
                        let a = *pointer;
                        let res = self.parse_expression(pointer);
                        let cpos = self.codepos_from_space(a, pointer, 1);
                        self.warning(CodeWarning::new_unnecessary_code(
                            cpos,
                            None,
                        ));
                        Ok(AST::Expression { expr: res?, position: cpos })
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
                    Ok(AST::Expression { expr: res?, position: cpos })
                }
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
        let term = self.parse_term(pointer)?;
        if self.match_token(pointer, TokenType::As)? {
            Ok(Expression::CastExpr {expr: Box::new(term), typ: self.parse_type(pointer)?})
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
                    node = Expression::BinaryOp { lhs: Box::new(node), op, rhs: Box::new(right) };
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
                TokenType::Star | TokenType::Slash => {
                    let op = self.advance(pointer).unwrap();
                    let right = self.parse_primary(pointer)?;
                    node = Expression::BinaryOp { lhs: Box::new(node), op, rhs: Box::new(right) };
                }
                _ => break,
            }
        }
        Ok(node)
    }

    fn parse_primary(&self, pointer: &mut usize) -> CodeResult<Expression> {
        if let Some(token) = self.advance(pointer) {
            match token.token_type {
                TokenType::NumberInt => Ok(Expression::IntNumber { value: token.content.parse().unwrap(), token }),
                TokenType::NumberFloat => Ok(Expression::FloatNumber { value: token.content.parse().unwrap(), token }),
                TokenType::Identifier => {
                    if self.match_next_token(pointer, TokenType::LParen)? {
                        self.parse_function_call(pointer)
                    } else {
                        Ok(Expression::Identifier(token))
                    }
                }
                TokenType::String => Ok(Expression::String(token)),
                TokenType::LParen => {
                    let expr = self.parse_expression(pointer)?;
                    if self.match_token(pointer, TokenType::RParen)? {
                        Ok(expr)
                    } else {
                        println!("LParen");
                        Err(CodeError::placeholder())
                    }
                }
                _ => Err(CodeError::new_unexpected_token_error(
                    self.previous(pointer).unwrap(),
                    TokenType::Expression,
                    Some(
                        "You may add a literal (number), string, variable, or a term here"
                            .to_string(),
                    ),
                )),
            }
        } else {
            Err(CodeError::missing_token_error(
                self.previous(pointer).unwrap(),
            ))
        }
    }

    fn parse_type(&self, pointer: &mut usize) -> CodeResult<Types> {
        let kind = 
            if self.match_token(pointer, TokenType::i32)? { Ok(TypesKind::I32) }
            else if self.match_token(pointer, TokenType::void)? { Ok(TypesKind::I32) }
            else if self.match_token(pointer, TokenType::Identifier)? { Ok(TypesKind::Struct {name: self.tokens[*pointer].content.clone() }) }
            else {Err(CodeError::not_a_type_error(&self.tokens[*pointer]))};
        
        Ok(Types::new(kind?, &self.tokens[*pointer]))
    }
}

#[derive(Debug, Hash, Copy, Clone)]
pub enum FunctionMode {
    Private,
    Export,
    Extern,
    Default,
}

#[derive(Debug, Hash, Clone)]
pub enum TypesKind {
    I32,
    F32,
    Void,
    Struct {name: String},
    Function {ret: Box<TypesKind>, params: Vec<TypesKind>}
}

#[derive(Debug, Hash, Clone)]
struct Types<'a> {
    kind: TypesKind,
    token: &'a Token
}

impl<'a> Types<'a> {
    pub fn new(kind: TypesKind, token: &'a Token) -> Self {
        Self {kind, token}
    }
}

#[derive(Debug, Hash)]
pub enum Expression<'a> {
    IntNumber {
        value: u64,
        token: &'a Token,
    },
    FloatNumber {
        value: u64,
        token: &'a Token,
    },
    Identifier(&'a Token),
    String(&'a Token),
    Type { typ: Types<'a>, token: &'a Token },
    // LHS, Opcode, RHS
    BinaryOp {lhs: Box<Expression<'a>>, op: &'a Token, rhs: Box<Expression<'a>>},
    // Expr, Type
    CastExpr { expr: Box<Expression<'a>>, typ: Types<'a> },
    FunctionCall { name: & 'a Token, arguments: Vec<Expression<'a>> },
}

#[derive(Debug, Hash)]
pub enum AST<'a> {
    Expression { expr: Expression<'a >, position: CodePosition },
    FunctionDef {
        name: & 'a Token,
        fmode: FunctionMode,
        ret: Types<'a>,
        args: Vec<(& 'a Token, Types<'a>)>,
        body: Vec<Box<AST<'a>>>,
    },
    VariableSet { name: & 'a Token, value: Expression<'a>, typ: Types<'a> },
    Return(Option<Expression<'a>>),
}