#lang racket

(require brag/support)
(require br-parser-tools/lex)
(require "terms.rkt")
(require "facts.rkt")
(require "grammar.rkt")

(define (parse-fact s)
  (let ([parse-tree (parse (tokenize (open-input-string s)))])
    (parse-tree-to-terms (syntax->datum parse-tree))))

(provide parse-fact)

(define (parse-tree-to-terms t)
  (match t
         [`(fact ,eq) (parse-tree-to-terms eq)]
         [`(expr ,e) (parse-tree-to-terms e)]
         [`(paren-expr ,_ ,e ,_) (parse-tree-to-terms e)]
         [`(expr_l1 ,e) (parse-tree-to-terms e)]
         [`(expr_l2 ,e) (parse-tree-to-terms e)]
         [`(expr_l3 ,e) (parse-tree-to-terms e)]
         [`(equality ,e1 ,_ ,e2)
           (Fact 'Eq (list
                       (parse-tree-to-terms e1)
                       (parse-tree-to-terms e2)))]
         [`(sum ,e1 ,_ ,e2)
           (BinOp op+ (parse-tree-to-terms e1) (parse-tree-to-terms e2))]
         [`(sub ,e1 ,_ ,e2)
           (BinOp op- (parse-tree-to-terms e1) (parse-tree-to-terms e2))]
         [`(prod ,e1 ,_ ,e2)
           (BinOp op* (parse-tree-to-terms e1) (parse-tree-to-terms e2))]
         [`(prod ,e1 ,e2)
           (BinOp op* (parse-tree-to-terms e1) (parse-tree-to-terms e2))]
         [`(div ,e1 ,_ ,e2)
           (BinOp op/ (parse-tree-to-terms e1) (parse-tree-to-terms e2))]
         [`(number ,n)
           (Number n)]
         [`(any_number ,_)
           (AnyNumber)]
         [`(number ,_ ,n)
           (Number (- n))]
         [`(variable ,v)
           (Variable v)]))

(define (tokenize ip)
  (port-count-lines! ip)
  (let* ([expr-lexer
           (lexer-src-pos
             ["(" (token 'LEFT_PAREN)]
             [")" (token 'RIGHT_PAREN)]
             ["=" (token 'REL_EQ)]
             ["+" (token 'OP_PLUS)]
             [(concatenation "-" (repetition 1 +inf.0 numeric))
              (token 'INTEGER (string->number lexeme))]
             ["-" (token 'OP_MINUS)]
             ["*" (token 'OP_TIMES)]
             ["/" (token 'OP_DIV)]
             ["?" (token 'ANY_NUMBER)]
             [(repetition 1 +inf.0 numeric)
              (token 'INTEGER (string->number lexeme))]
             [(repetition 1 +inf.0 alphabetic)
              (token 'VARIABLE lexeme)]
             [whitespace
              (token 'WHITESPACE lexeme #:skip? #t)]
             )]
         [next-token (lambda () (expr-lexer ip))])
    next-token))
