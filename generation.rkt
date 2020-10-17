; Generates random problems.

#lang racket

(require racket/random)
(require racket/engine)
(require "terms.rkt")
(require "facts.rkt")
(require "solver.rkt")
(require "tactics.rkt")
(require "serialize.rkt")

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
      (list (assumption (generate-equation (list variable))))
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

; Constant timeout for the problem generation place below, in ms.
(define timeout 20000)

; A place task that executes an infinite loop of generating
; and solving problems.
(define (problem-generation-place channel)
  (let* ([problem (generate-problem)]
         [e (engine (lambda (_) (solve-problem
                                  problem
                                  s:all
                                  (prune:keep-smallest-k 50)
                                  20)))]
         [success? (engine-run timeout e)])
    (if success?
      (begin
        (printf "Solver succeeded!\n")
        (place-channel-put
          channel
          (to-jsexpr 
            (hash
              'problem problem
              'solution (engine-result e))))
        (problem-generation-place channel))
        (begin
          (printf "Solver timed out...\n")
          (problem-generation-place channel)))))

; Generates and solves problems in parallel, saving them from time
; to time in `out-path` in JSON format.
(define (generate-problems-job n-threads out-path)
  (with-handlers
    () ; TODO: handle Control-C
    (letrec ([places (map (lambda (_) (place ch (problem-generation-place ch)))
                          (range n-threads))]
             [loop (lambda (all i)
                     (define result (place-channel-get
                                      (list-ref places 
                                                (remainder i n-threads))))
                     (if (eq? 0 (remainder i 10))
                       (begin
                         (printf "Saving ~a problems to ~a.\n"
                                 (length all) out-path)
                         (call-with-output-file out-path
                           (lambda (out) (to-json all out))))
                       void)
                     (loop (cons result all) (+ 1 i)))])
      (loop empty 0))))

(provide
  generate-problem
  generate-problems-job)
