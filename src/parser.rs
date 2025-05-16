use std::collections::VecDeque;
use std::str::Chars;

#[derive(Debug, Clone, PartialEq)]
enum Token {
    I32,
    F32,
    Struct,
    Identifier(String),
    LParen,
    RParen,
    LBrace,
    RBrace,
    Semicolon,
    Comma,
    Asterisk,
    Ref,
    And,
    Or,
    Not,
    Pipe,
    Eq,
    DEq,
    NEq,
    Gt,
    Lt,
    Gte,
    Lte,
    Return,
    If,
    Else,
    For,
    While,
    Break,
    Void,
    EOF,
    NumberInt(i64),
    NumberFloat(f64),
    Slash,
    Minus,
    Plus,
}

#[derive(Debug, Clone)]
pub enum Expression {
    Identifier(String),
    IntNumber(i64),
    FloatNumber(f64),
    Binary {
        op: BinaryOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    FunctionCall {
        name: String,
        arguments: Vec<Expression>
    },
}

#[derive(Debug, Clone)]
pub enum BinaryOp {
    Eq,
    Neq,
    Gt,
    Lt,
    Gte,
    Lte,
    Not,
    Add,
    Sub,
    Div,
    Mul,
    And,
    Or
}

impl From<&Token> for Option<BinaryOp> {
    fn from(value: &Token) -> Self {
        match value {
            Token::Asterisk => Some(BinaryOp::Mul),
            Token::Minus => Some(BinaryOp::Sub),
            Token::Plus => Some(BinaryOp::Add),
            Token::Slash => Some(BinaryOp::Div),
            Token::And => Some(BinaryOp::And),
            Token::Or => Some(BinaryOp::Or),
            Token::Not => Some(BinaryOp::Not),
            Token::DEq => Some(BinaryOp::Eq),
            Token::NEq => Some(BinaryOp::Neq),
            Token::Gt => Some(BinaryOp::Gt),
            Token::Lt => Some(BinaryOp::Lt),
            Token::Gte => Some(BinaryOp::Gte),
            Token::Lte => Some(BinaryOp::Lte),
            _ => None
        }
    }
}

#[derive(Clone)]
#[derive(Debug)]
pub enum Types {
    I32,
    F32,
    Void,
    Struct {name: String}
}

#[derive(Debug, Clone)]
pub enum AST {
    FunctionDef {
        return_type: Types,
        name: String,
        params: Vec<(Types, String)>,
        body: Option<Vec<AST>>
    },
    VarDef {
        var_type: Types,
        name: String,
        value: Option<Expression>
    },
    Expr {
        expr: Expression
    },
    Return(Expression)
}

impl From<Expression> for AST {
    fn from(value: Expression) -> Self {
        AST::Expr {expr: value}
    }
}

fn token_is_type_like(tok: &Token) -> bool {
    match tok {
        Token::I32 => true,
        Token::F32 => true,
        Token::Identifier(_) => true,
        _ => false
    }
}

struct Lexer<'a> {
    input: Chars<'a>,
    current: Option<char>,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        let mut lexer = Lexer {
            input: src.chars(),
            current: None,
        };
        lexer.bump();
        lexer
    }

    fn bump(&mut self) {
        self.current = self.input.next();
    }

    fn next_token(&mut self) -> Token {
        while let Some(c) = self.current {
            match c {
                ' ' | '\n' | '\t' | '\r' => self.bump(),
                '(' => {
                    self.bump();
                    return Token::LParen;
                }
                ')' => {
                    self.bump();
                    return Token::RParen;
                }
                '{' => {
                    self.bump();
                    return Token::LBrace;
                }
                '}' => {
                    self.bump();
                    return Token::RBrace;
                }
                ';' => {
                    self.bump();
                    return Token::Semicolon;
                }
                ',' => {
                    self.bump();
                    return Token::Comma;
                }
                '*' => {
                    self.bump();
                    return Token::Asterisk;
                }
                '+' => {
                    self.bump();
                    return Token::Plus;
                }
                '-' => {
                    self.bump();
                    return Token::Minus;
                }
                '/' => {
                    self.bump();
                    return Token::Slash;
                }
                '&' => {
                    self.bump();
                    if self.current == Some('&') {
                        self.bump();
                        return Token::And;
                    } else {
                        return Token::Ref;
                    }
                }
                '=' => {
                    self.bump();
                    if self.current == Some('=') {
                        self.bump();
                        return Token::DEq;
                    } else {
                        return Token::Eq;
                    }
                }
                '>' => {
                    self.bump();
                    if self.current == Some('=') {
                        self.bump();
                        return Token::Gte;
                    } else {
                        return Token::Gt;
                    }
                }
                '<' => {
                    self.bump();
                    if self.current == Some('=') {
                        self.bump();
                        return Token::Lte;
                    } else {
                        return Token::Lt;
                    }
                }
                '!' => {
                    self.bump();
                    if self.current == Some('=') {
                        self.bump();
                        return Token::NEq;
                    } else {
                        return Token::Not;
                    }
                }
                '|' => {
                    self.bump();
                    if self.current == Some('|') {
                        self.bump();
                        return Token::Or;
                    } else {
                        return Token::Pipe;
                    }
                }
                '0'..='9' => return self.lex_number(),
                'a'..='z' | 'A'..='Z' | '_' => return self.lex_identifier(),
                _ => panic!("Unexpected character: {}", c),
            }
        }
        Token::EOF
    }

    fn lex_number(&mut self) -> Token {
        let mut num = String::new();
        let mut is_float = false;

        while let Some(c) = self.current {
            if c.is_ascii_digit() {
                num.push(c);
                self.bump();
            } else if c == '.' {
                if is_float {
                    panic!("Unexpected second '.' in number");
                }
                is_float = true;
                num.push(c);
                self.bump();
            } else {
                break;
            }
        }

        if is_float {
            let val: f64 = num.parse().expect("Invalid float literal");
            Token::NumberFloat(val)
        } else {
            let val: i64 = num.parse().expect("Invalid integer literal");
            Token::NumberInt(val)
        }
    }

    fn lex_identifier(&mut self) -> Token {
        let mut ident = String::new();
        while let Some(c) = self.current {
            if c.is_alphanumeric() || c == '_' {
                ident.push(c);
                self.bump();
            } else {
                break;
            }
        }

        match ident.as_str() {
            "struct" => Token::Struct,
            "i32" => Token::I32,
            "f32" => Token::F32,
            "return" => Token::Return,
            "break" => Token::Break,
            "while" => Token::While,
            "if" => Token::If,
            "else" => Token::Else,
            "for" => Token::For,
            "void" => Token::Void,
            _ => Token::Identifier(ident),
        }
    }
}


struct Parser<'a> {
    lexer: Lexer<'a>,
    lookahead: Token,
    buffer: VecDeque<Token>,
}

impl<'a> Parser<'a> {
    fn new(mut lexer: Lexer<'a>) -> Self {
        let lookahead = lexer.next_token();
        Parser {
            lexer,
            lookahead,
            buffer: VecDeque::new(),
        }
    }

    fn bump(&mut self) {
        if let Some(tok) = self.buffer.pop_front() {
            self.lookahead = tok;
        } else {
            self.lookahead = self.lexer.next_token();
        }
    }


    fn peek(&mut self, n: usize) -> &Token {
        while self.buffer.len() < n {
            let tok = self.lexer.next_token();
            self.buffer.push_back(tok);
        }
        &self.buffer[n - 1]
    }


    fn expect(&mut self, expected: Token) {
        if self.lookahead == expected {
            self.bump();
        } else {
            panic!("Expected {:?}, found {:?}", expected, self.lookahead);
        }
    }

    fn parse(&mut self) -> Vec<AST> {
        let mut items = Vec::new();
        while self.lookahead != Token::EOF {
            items.push(self.parse_function_or_decl());
        }
        items
    }

    fn parse_type(&mut self) -> Types {
        match self.lookahead.clone() {
            Token::I32 => {
                self.bump();
                Types::I32
            }
            Token::F32 => {
                self.bump();
                Types::F32
            }
            Token::Void => {
                self.bump();
                Types::F32
            }
            Token::Identifier(name) => {
                self.bump();
                Types::Struct {name}
            }
            _ => panic!("Expected type"),
        }
    }

    fn parse_identifier(&mut self) -> String {
        if let Token::Identifier(name) = &self.lookahead {
            let id = name.clone();
            self.bump();
            id
        } else {
            panic!("Expected identifier, got {:?}", self.lookahead);
        }
    }

    fn parse_function_or_decl(&mut self) -> AST {
        let var_type = self.parse_type();
        let name = self.parse_identifier();

        if self.lookahead == Token::LParen {
            self.bump(); // (
            let mut params = Vec::new();
            if self.lookahead != Token::RParen {
                loop {
                    let ptype = self.parse_type();
                    let pname = self.parse_identifier();
                    params.push((ptype, pname));
                    if self.lookahead == Token::Comma {
                        self.bump();
                    } else {
                        break;
                    }
                }
            }

            self.expect(Token::RParen);
            if self.lookahead == Token::Semicolon {
                self.bump();
                return AST::FunctionDef {
                    return_type: var_type,
                    name,
                    params,
                    body: None
                }
            }
            self.expect(Token::LBrace);

            let mut body = Vec::new();
            while self.lookahead != Token::RBrace {
                let statement = self.parse_statement();
                println!("{:?}", statement);
                body.push(statement);
            }
            self.expect(Token::RBrace);

            AST::FunctionDef {
                return_type: var_type,
                name,
                params,
                body: Some(body),
            }
        } else {
            panic!("Only function definitions are allowed at top-level")
        }
    }

    fn parse_statement(&mut self) -> AST {
        if token_is_type_like(&self.lookahead) && matches!(self.peek(2), Token::Eq | Token::Semicolon) {
            return self.parse_var_def()
        }
        match &self.lookahead {
            Token::Identifier(_) => {
                let expr = self.parse_expr();
                self.expect(Token::Semicolon);
                expr.into()
            }
            Token::Return => {
                self.bump();
                let expr = self.parse_expr();
                self.expect(Token::Semicolon);
                AST::Return(expr)
            }
            _ => panic!("Unexpected token in function body: {:?}", self.lookahead),
        }
    }

    fn parse_expr(&mut self) -> Expression {
        let mut left = self.real_parse_primary();

        let op = <&Token as Into<Option<BinaryOp>>>::into((&self.lookahead).into());
        if op.is_none() {return left}
        let op = op.unwrap();
        self.bump();

        let right = self.parse_expr();
        left = Expression::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        };

        left
    }

    fn parse_fcall(&mut self, name: String) -> Expression {
        self.expect(Token::LParen);
        let mut arguments = vec![];
        if self.lookahead != Token::RParen {
            loop {
                arguments.push(self.parse_expr());
                if self.lookahead == Token::Comma {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.expect(Token::RParen);
        Expression::FunctionCall {name, arguments}
    }

    fn real_parse_primary(&mut self) -> Expression {
        let p = self.parse_primary();
        if self.lookahead == Token::LParen { match p {
            Expression::Identifier(name) => {
                self.parse_fcall(name)
            }
            _ => {p} }
        }
        else {
            p
        }
    }

    fn parse_primary(&mut self) -> Expression {
        match &self.lookahead {
            Token::Identifier(name) => {
                let id = name.clone();
                self.bump();
                Expression::Identifier(id)
            }
            Token::NumberInt(int) => {
                let int = int.clone();
                self.bump();
                Expression::IntNumber(int)
            }
            Token::NumberFloat(int) => {
                let int = int.clone();
                self.bump();
                Expression::FloatNumber(int)
            }
            _ => panic!("Expected primary expression, found {:?}", self.lookahead),
        }
    }

    fn parse_var_def(&mut self) -> AST {
        let var_type = self.parse_type();
        let name = self.parse_identifier();
        if self.lookahead == Token::Semicolon {
            self.expect(Token::Semicolon);
            return AST::VarDef { var_type, name, value: None }
        }
        self.expect(Token::Eq);
        let value = self.parse_expr();
        self.expect(Token::Semicolon);
        AST::VarDef { var_type, name, value: Some(value) }
    }
}

pub fn analyze(src: &str) {
    let mut lexer = Lexer::new(src);
    let mut tok = Token::Semicolon;
    while tok != Token::EOF {
        tok = lexer.next_token();
        print!("{:?}\n", tok);
    }
}

pub fn parse_file_source(source: &str) -> Vec<AST> {
    let lexer = Lexer::new(source);
    let mut parser = Parser::new(lexer);
    parser.parse()
}