; Generates random problems.

#lang racket

(require racket/random)
(require "terms.rkt")
(require "facts.rkt")
(require "solver.rkt")

(define (choice l) (car (random-sample l 1)))
(define (choice-p l [c 0.0] [r #f])
  (let ([r (or r (random))]
        [x (caar l)]
        [p (cdar l)]
        [tail (cdr l)])
    (if (<= r (+ p c)) x (choice-p tail (+ p c) r))))

(define (generate-problem)
  (match (choice (list 'equation 'evaluation))
    [(== 'equation) (generate-equation-problem)]
    [(== 'evaluation) (generate-equation-problem)]))

(define (generate-equation-problem)
  (let ([variable (string (choice "abcdefghijklmnopqrstuvwxyz"))])
    (Problem 
      (list (generate-equation (list variable)))
      (list (Predicate 'Eq (list
                             (Variable variable)
                             (AnyNumber)))))))

(define (generate-equation variables)
  (generate-predicate (list 'Eq) variables))

(define (generate-predicate types variables)
  (Predicate (choice types)
    (list
      (generate-expression variables)
      (generate-expression variables))))

(define (generate-expression variables [p-binop 0.5] [decay 0.75])
  (match (choice-p (list (cons 'Number (* 0.6 (- 1 p-binop)))
                         (cons 'Variable (* 0.2 (- 1 p-binop)))
                         (cons 'VariableCoeff (* 0.1 (- 1 p-binop)))
                         (cons 'UnOp (* 0.1 (- 1 p-binop)))
                         (cons 'BinOp p-binop)))
    [(== 'Number)
     (Number (choice (range -10 11)))]
    [(== 'Variable)
     (Variable (choice variables))]
    [(== 'VariableCoeff)
     (BinOp op*
            (Number (choice (range -10 11)))
            (Variable (choice variables)))]
    [(== 'UnOp)
     (UnOp op- (generate-expression variables p-binop))]
    [(== 'BinOp)
     (BinOp (choice (list op+ op- op* op/))
            (generate-expression variables (* p-binop decay) decay)
            (generate-expression variables (* p-binop decay) decay))]
    ))

(provide generate-problem)
