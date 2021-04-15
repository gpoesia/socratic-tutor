; Generates random problems.

#lang algebraic/racket/base

(require racket/random)
(require racket/format)
(require (rename-in racket/list [permutations list-permutations]))
(require racket/engine)
(require racket/place)
(require racket/match)
(require math/number-theory)
(require "terms.rkt")
(require "facts.rkt")
(require "solver.rkt")
(require "equations.rkt")

(define (choice l) (car (random-sample l 1)))
(define (choice-p l [c 0.0] [r #f])
  (let ([r (or r (random))]
        [x (caar l)]
        [p (cdar l)]
        [tail (cdr l)])
    (if (<= r (+ p c)) x (choice-p tail (+ p c) r))))

; Returns #t with probability p, #f otherwise.
(define (flip-coin p) (< (random) p))

; Returns a number between -10 and 10
(define (random-small-integer) (choice (range -10 11)))

; Returns a number between -10 and 10 that is not zero.
(define (random-nonzero-small-integer)
  (* (choice (range 1 11)) (choice (list -1 1))))

; Generates a term that is equivalent to the given number.
(define (random-eqv-number n)
  (match (choice '(+ - * /))
    [(== '+) (let ([m (random-small-integer)])
               (BinOp op+ (Number m) (Number (- n m))))]
    [(== '-) (let ([m (random-small-integer)])
               (BinOp op- (Number (+ n m)) (Number m)))]
    [(== '/) (let ([m (random-nonzero-small-integer)])
               (BinOp op/ (Number (* n m)) (Number m)))]
    [(== '*) (let ([f (if (eq? 0 n) (random-nonzero-small-integer)
                                    (choice (divisors n)))])
               (BinOp op* (Number (/ n f)) (Number f)))]))

; Generates either a Number of a Number times a variable.
(define (random-atomic-term vars)
  (match (choice '(n v))
    [(== 'n) (Number (random-small-integer))]
    [(== 'v) (if (flip-coin 0.5)
               (BinOp op* (Number (random-small-integer))
                          (Variable (choice vars)))
               (Variable (choice vars)))]))

; Finds a variable in ans that is equal to the given number.
(define (find-var-by-value ans n)
  (findf (lambda (var-value) (equal? (cdr var-value) n)) ans))

; Generates a random term that is equivalent to t.
; This is the core of problem generation. The idea is to flip
; a coin with probability p and make local changes to the term,
; like applying associativity, commutativity, or replacing a number
; by an expression that evaluates to that number.
(define (random-eqv-term t p ans)
  ((function
     [(Number n)
      #:if (flip-coin p)
      (random-eqv-number n)]
     [(Number n)
      #:if (and
             (flip-coin p)
             (find-var-by-value ans n))
      (Variable (car (find-var-by-value (shuffle ans) n)))]
     [(Variable v)
      #:if (flip-coin p)
      (BinOp op* (random-eqv-number 1) (Variable v))]
     [(UnOp t)
      #:if (flip-coin p)
      (BinOp op* (random-eqv-number -1)
             (random-eqv-term t p ans))]
     [(BinOp (op #:if (is-commutative? op)) a b)
      #:if (flip-coin p)
        (BinOp op (random-eqv-term b p ans)
                  (random-eqv-term a p ans))]
     [(BinOp op1 (BinOp op2 a b) c)
      #:if (and (flip-coin p) (is-associative? op1 op2))
      (BinOp op1 (random-eqv-term a p  ans)
                 (BinOp op2 (random-eqv-term b p ans)
                            (random-eqv-term c p ans)))]
     ; Eval constant operation.
     [(BinOp op (Number n1) (Number n2))
      #:if (flip-coin p)
      (Number (compute-bin-op op n1 n2))]
     ; Flip equality.
     [(Predicate 'Eq (a b))
      #:if (flip-coin p)
      (Predicate 'Eq (list (random-eqv-term b p ans)
                           (random-eqv-term a p ans)))]
     ; Perturb both sides.
     [(Predicate 'Eq (a b))
      #:if (flip-coin p)
      (Predicate 'Eq (list (random-eqv-term a p ans)
                           (random-eqv-term b p ans)))]
     ; Perform same operation on both sides.
     [(Predicate 'Eq (a b))
      #:if (flip-coin p)
      (match (choice '(+ - *))
        [(== '+)
         (let ([t (random-atomic-term (map car ans))])
          (Predicate 'Eq (list (BinOp op+ a t) (BinOp op+ b t))))]
        [(== '-)
         (let ([t (random-atomic-term (map car ans))])
          (Predicate 'Eq (list (BinOp op- a t) (BinOp op- b t))))]
        [(== '*)
         (let ([n (random-small-integer)])
          (Predicate 'Eq (list (BinOp op* a (Number n)) (BinOp op- b (Number n)))))])]
     [_ t]) t))

(define (repeat-perturb t n p ans)
  (if (eq? n 0) t
    (repeat-perturb (random-eqv-term t p ans) (- n 1) p ans)))

(define (generate-problem [max-difficulty 15] [max-variables 1])
  (let* ([variables (map ~a (take (shuffle (string->list "abcdefghijklmnopqrstuvwxyz"))
                                  (choice (range 1 (+ 1 max-variables)))))]
         [answer (map (lambda (v) (cons v (random-small-integer))) variables)]
         [goals (map (lambda (v) (Predicate 'Eq (list (Variable v)
                                                      (AnyNumber))))
                     variables)]
         [base-facts
           (map (lambda (var-value)
                  (Predicate 'Eq (list (Variable (car var-value))
                                       (Number (cdr var-value)))))
                answer)])
    (Problem
      (map (lambda (t) (assumption (repeat-perturb t (random 1 max-difficulty) 0.3 answer))) base-facts)
      goals)))

(provide
  generate-problem)
