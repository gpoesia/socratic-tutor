; Axioms and tactics available in the "Fraction" environment.
#lang algebraic/racket/base

(require "terms.rkt")

(require racket/list)
(require racket/string)
(require racket/function)
(require racket/match)
(require "debug.rkt")
(require "facts.rkt")
(require "solver.rkt")

; ==============================
; ========   Axioms  ===========
; ==============================
;;; A . cancel common factor ⇒ (2*3) / (2*4) = ¾
;;; B . expand a number into two factors ⇒  24 = 2*12
;;; C.  convert a number into fraction by dividing one ⇒  3 = 3/1
;;; D.  a number divided by 1 is itself ⇒  25/1 = 25
;;; E.  multiply a fraction by a scaling factor ⇒  1/1 = 2/2
;;; F.  merge two fractions ⇒  2/2 + 3/2 = (2+3)/2
;;; G.  A number over a multiple of itself should be simplied ⇒ 10/5 = 2
;;; H.  Evaluates a binary operation on numbers, except for fractions
;;; I. Commutativity

; A. cancel out common factor
; Possible if factor exists in both numerator and demonimator
(define fd:cancel-common-factor?
  (function*
   [( (BinOp op n1 n2) f)
    #:if (and (factor-appears-in? n1 f) (factor-appears-in? n2 f))
    #t]
   [_ #f]))

(define fd:cancel-common-factor
  (phi* ((BinOp op n1 n2) f)
        (BinOp op/ (remove-factor-from n1 f) (remove-factor-from n2 f))))

(define factor-appears-in?
  (function*
   [( (Number n) f)
    #:if (eq? n f)
    #t]
   [((BinOp op n1 n2) f)
    #:if (and (eq? op op*) (or (factor-appears-in? n1 f) (factor-appears-in? n2 f)))
    #t]
   [_ #f]))

(define remove-factor-from
  (function*
   [((Number n) f) #:if (eq? n f) (Number 1)]
   [((BinOp op n1 n2) f)
    (let ([trimmed_n1 (remove-factor-from n1 f)]
          [trimmed_n2 (remove-factor-from n2 f)])
      (cond
        [trimmed_n1 (BinOp op trimmed_n1 n2)]
        [trimmed_n2 (BinOp op n1 trimmed_n2)]
        [else #f]
        ))
    ]
   [_ #f]))

; B. expand a number into two factors
(define fd:factorize?
  (function*
   [((Number N) k)
    #:if (divisible? N k)
    #t]
   [_ #f]))

(define fd:factorize
  (phi* ((Number N) k)
        (BinOp op* (Number (/ N k)) (Number k))))

(define (divisible? n x)
  (zero? (remainder n x)))

; C.  convert a number into fraction by dividing one ⇒  3 = 3/1
(define fd:convert-into-fraction?
  (function
   [(Number N) #t]
   [_  #f]))

(define fd:convert-into-fraction
  (phi (Number N)
       (BinOp op/ (Number N) (Number 1))))

; D. a number divided by 1 is itself ⇒  25/1 = 25
;    a number multiplied by 1 is also itself
(define fd:mul-one?
  (function
   [(BinOp (op #:if (eq? op op*)) (Number 1) t) #t]
   [(BinOp (op #:if (eq? op op*)) t (Number 1)) #t]
   [(BinOp (op #:if (eq? op op/)) t (Number 1)) #t]
   [t #f]))

(define fd:mul-one
  (function
   [(BinOp (op #:if (eq? op op*)) (Number 1) t) t]
   [(BinOp (op #:if (eq? op op*)) t (Number 1)) t]
   [(BinOp (op #:if (eq? op op/)) t (Number 1)) t]
   [t #f]))

; E. multiply a fraction by a scaling factor ⇒  1/1 = 2/2
(define fd:mul-scaling-factor?
  (function*
   [( (BinOp (op #:if (eq? op op/)) _ _) scale) #t]
   [_ #f]))

(define fd:mul-scaling-factor
  (phi* ((BinOp (op #:if (eq? op op/)) n1 n2) scale)
        (BinOp op/ (BinOp op* (Number scale) n1) (BinOp op* (Number scale) n2))))

; F.  merge two fractions ⇒  2/2 +/- 3/2 = (2+/-3)/2
(define fd:merge-two-fractions?
  (function
   [ (BinOp op0 (BinOp op1 n1 n2) (BinOp op2 m1 m2))
     #:if (and (or (eq? op0 op+) (eq? op0 op-)) (eq? op1 op/) (eq? op2 op/) (equal? n2 m2))
     #t]
   [_ #f]))


(define fd:merge-two-fractions
  (phi (BinOp op0 (BinOp op1 n1 n2) (BinOp op2 m1 m2))
       (BinOp op/ (BinOp op0 n1 m1) n2)))


; H. Evaluates binary operations, except for division with one exception:
; G. A number over a multiple of itself should be simplied 10/5 = 2

(define fd:binop-eval?
  (function
   [(BinOp op (Number n1) (Number n2))
    #:if (not (eq? op op/))
    #t]
   [(BinOp op (Number n1) (Number n2))
    #:if (divisible? n1 n2)
    #t]
   [_ #f]))

(define fd:binop-eval
  (phi (BinOp op (Number n1) (Number n2))
       (Number (compute-bin-op op n1 n2))))

; I. Commutativity

(define fd:commutativity?
  (function
    [(BinOp op _ _) (is-commutative? op)]
    [_ #f]))

(define fd:commutativity
  (phi (BinOp op l r) (BinOp op r l)))


; ==============================
; ========   Tactics ===========
; ==============================

;;; A. Try cancel common factor for every fraction; Using factors defined in the `primes` list + set of number appearing in the expression
;;; B. Try expanding every number into two factors; Again, using factors in the `primes` list + set of number appearing in the expression
;;; C. Try converting every  number into a fraction
;;; D. Try multiplying every fraction by a scalar factor, using numbers in the `primes` list + set of number appearing in the expression
;;; E. Try merge two fractions if they share the same denominator
;;; F. Evaluate binary operation except for division
;;; G. Commute every possible pairs

(define MAX-SIZE 20)

; Meta-tactic that applies a simple term-level transform pair
; to all new facts, in all terms that satisfy the given predicate.
(define-syntax-rule (local-rewrite-tactic name predicate transform)
  (define (name unmet-goals old-facts new-facts)
    (apply append
           (map (lambda (f)
                  (log-debug "fact ~a\n" (format-term (Fact-term f)))
                  (let ([indices (filter-subterms (Fact-term f) predicate)])
                    (log-debug "indices ~a\n" indices)
                    (map (lambda (i)
                           (let ([rewritten (rewrite-subterm (Fact-term f) transform i)])
                             (log-debug "~a rewrote ~a => ~a\n"
                                        #(name)
                                        (format-term (Fact-term f))
                                        (format-term rewritten))
                             (if rewritten
                                 (fact rewritten
                                       (FactProof
                                        transform
                                        (list (FactId (Fact-id f)) i))) ;
                                 #f
                                 )))
                         indices)))
                new-facts))))

; Meta-tactic that applies at term-level
; Difference to `local-rewrite-tactic` is the additional param `context`
; For example, when applying the tactic `cancle-common-factor`. `context` would be the choices for factors
(define-syntax-rule (local-rewrite-tactic-w-context name predicate transform context)
  (define (name unmet-goals old-facts new-facts)
    (apply append
           (map (lambda (f)
                  (log-debug "fact-term ~a \n" (Fact-term f))
                  (let ([context-options (context (Fact-term f))])
                    (apply append
                           (map (lambda (c)
                                  (let ([indices (filter-subterms-w-context (Fact-term f) predicate c)])
                                    (map (lambda (i)
                                           (let ([rewritten (rewrite-subterm-w-context (Fact-term f) transform i c)])
                                             (log-debug "~a rewrote ~a => ~a\n"
                                                        #(name)
                                                        (format-term (Fact-term f))
                                                        (format-term rewritten))
                                             (if rewritten
                                                 (fact rewritten
                                                       (FactProof
                                                        transform
                                                        (list (FactId (Fact-id f)) i))) ;need toincorportate context into factid?
                                                 #f
                                                 )))
                                         indices)))
                                context-options))))
                new-facts))))


;A. Try cancel common factor for every fraction; Using factors defined in the `primes` list + set of number appearing in the expression
;List all numbers appearing in the expression
(define enumerate-all-numbers-in-expression
  (function
   [(Number a) (list a)]
   [(BinOp op t1 t2) (append (enumerate-all-numbers-in-expression t1) (enumerate-all-numbers-in-expression t2))]
   ))
;List all numbers appearing in the expression + numbers in the `primes` list
(define (enumerate-all-numbers f)
  (log-debug "enumerate-all-numbers ~a \n" f)
  (remove-duplicates (append (enumerate-all-numbers-in-expression f)  primes) );
  )

(local-rewrite-tactic-w-context fdt:cancel-common-factor fd:cancel-common-factor? fd:cancel-common-factor enumerate-all-numbers)

; B. Try expanding every number into two factors; Again, using factors in the `primes` list + set of number appearing in the expression
(local-rewrite-tactic-w-context fdt:factorize fd:factorize? fd:factorize enumerate-all-numbers)

;C. Try converting every number into a fraction
(local-rewrite-tactic fdt:convert-into-fraction fd:convert-into-fraction? fd:convert-into-fraction)

; D. Try multiplying every fraction by a scalar factor; Again, using factors in the `primes` list + set of number appearing in the expression
(local-rewrite-tactic-w-context fdt:mul-scaling-factor fd:mul-scaling-factor? fd:mul-scaling-factor enumerate-all-numbers)

; E. Try merge two fractions if they share the same denominator
(local-rewrite-tactic fdt:merge-two-fractions fd:merge-two-fractions? fd:merge-two-fractions)

; F. Evaluate binary operation except for division
(local-rewrite-tactic fdt:binop-eval fd:binop-eval? fd:binop-eval)

; G. Try commute every binary operation
(local-rewrite-tactic fdt:commutativity fd:commutativity? fd:commutativity)


(define (combine-tactics tactics)
  (lambda (unmet-goals old-facts new-facts)
    (apply append
           (map (lambda (t) (t unmet-goals old-facts new-facts))
                tactics))))


; Applies all tactics.
(define fdt:all (combine-tactics
                 (list
                  fdt:cancel-common-factor
                  fdt:factorize
                  fdt:convert-into-fraction
                  fdt:mul-scaling-factor
                  fdt:merge-two-fractions
                  fdt:binop-eval
                  fdt:commutativity
                  )))

; Domain function: given a node, lists all child nodes.
(define (d:fraction facts)
  ; Avoid huge equations.
  (if (> (term-size (Fact-term (last facts))) MAX-SIZE)
      empty
      (filter (lambda (f) (not (member f facts fact-terms-equal?)))
              (fdt:all #f empty (list (last facts))))))

; Generates a random fraction problem
(define (generate-fraction-problem [max-number-terms 3])
  (let* ([number-of-terms (random 1 (+ 1 max-number-terms))]
         [fractions (map (lambda (_) (generate-fraction))(range number-of-terms))])
    (Problem
     (list (assumption (FractionExpression (list-to-sum fractions))))
     (list (Number)))
    ))

; Convert a list of single elements into a sum through BinOp
(define (flip-coin p) (< (random) p))

(define (list-to-sum fractions)
(let ([op (if (flip-coin 0.5) op+ op-)])
  (if (eq? (length fractions) 1)
      (car fractions)
      (BinOp op (car fractions) (list-to-sum (cdr fractions))))))

; Define the list of prime factors allowed in the generated exercises
(define primes '(1 2 3 5 7))

; Generate a single fraction
(define (generate-fraction [max-number-primes-factors 7])
  (let* (
         [len-primes-numerator (random 1 max-number-primes-factors)]
         [len-primes-denominator (random 1 max-number-primes-factors)]
         [list-primes-numerator (map (lambda (_) (list-ref primes (random 0 (length primes)))) (range len-primes-numerator))]
         [list-primes-denominator (map (lambda (_) (list-ref primes (random 0 (length primes))))(range len-primes-denominator))]
         [numerator (Number (foldl * 1 list-primes-numerator))]
         [denominator (Number (foldl * 1 list-primes-denominator))])
    (BinOp op/ numerator denominator)))


(define (is-fraction-simplified? f g)
  (let ([x (FractionExpression-elems (Fact-term f))])
    (or (Number? x) (is-simple-fraction? x) )
    ))

; a simple fraction is of the form A / B
; where B is not 1, and gcd(A, B) = 1
(define is-simple-fraction?
  (function
   [(BinOp op (Number n) (Number d))
    #:if (and (eq? op/ op) (not (eq? 1 d))  (eq? 1 (gcd n d)))
    #t]
   [_ #f]))

(define (gcd a b)
  (cond
    [(> a b) (gcd b (- a b))]
    [(< a b) (gcd a (- b a))]
    [else a])
  )

(provide
 fd:cancel-common-factor
 fd:factorize
 fd:merge-two-fractions
 fd:mul-scaling-factor
 fd:binop-eval
 fd:convert-into-fraction
 fd:commutativity
 d:fraction
 generate-fraction-problem
 is-fraction-simplified?
 )
