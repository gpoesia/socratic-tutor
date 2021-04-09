; Axioms and tactics available in the "Sorting" environment.
#lang algebraic/racket/base

(require "terms.rkt")

(require racket/list)
(require racket/string)
(require racket/function)
(require racket/match)

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
;;; G.  A number over itself is 1
;;; H. Evaluates a binary operation on numbers, except for fractions


; A. cancel out common factor
; Possible if factor exists in both numerator and demonimator
(define cancel-common-factor?
  (function*
    [( (BinOp op n1 n2) (Number f))
     #:if (and (factor-appears-in n1 f) (factor-appears-in n2 f))
     #t]
    [_ #f]))

(define factor-appears-in 
  (function*
    [( (Number n) (Number f)) 
     #:if (eq? n f)
     #t]
    [((BinOp op n1 n2) (Number f))
     #:if (and (eq? op op*) (or (factor-appears-in n1 f) (factor-appears-in n2 f)))
     #t]
    [_ #f]))
  

; B. expand a number into two factors
(define (divisible? n x)
  (zero? (remainder n x))) 


(define factorize?
  (function*
    [((Number N) (Number k))
     #:if (divisible? N k)
     #t]
    [_ #f]))

(define factorize
  (phi* ((Number N) (Number k))
       (BinOp op* (/ N k) k)))

; C.  convert a number into fraction by dividing one ⇒  3 = 3/1
(define convert-into-fraction?
  (function
    [(Number N)
     #t]
    [_ #f]))

(define convert-into-fraction
  (phi (Number N) 
      (BinOp op/ N 1)))
      

; D. a number divided by 1 is itself ⇒  25/1 = 25
;    a number multiplied by 1 is also itself 
(define a:mul-one?
  (function
    [(BinOp (op #:if (eq? op op*)) (Number 1) t) #t]
    [(BinOp (op #:if (eq? op op*)) t (Number 1)) #t]
    [(BinOp (op #:if (eq? op op/)) t (Number 1)) #t]
    [t #f]))

(define a:mul-one
  (function
    [(BinOp (op #:if (eq? op op*)) (Number 1) t) t]
    [(BinOp (op #:if (eq? op op*)) t (Number 1)) t]
    [(BinOp (op #:if (eq? op op/)) t (Number 1)) t]
    [t #f]))

; E. multiply a fraction by a scaling factor ⇒  1/1 = 2/2
(define a:mul-scaling-factor?
  (function*
    [( (BinOp (op #:if (eq? op op/)) _ _) (Number scale)) #t]
    [_ #f]))

(define mul-scaling-factor
  (phi* ((BinOp (op #:if (eq? op op/)) n1 n2) (Number scale))
      (BinOp op (BinOp op* scale n1) (BinOp op* scale n2))))  
      ;;; or it could be (BinOp op (* scale n1) (* scale n2))))
  
; F.  merge two fractions ⇒  2/2 + 3/2 = (2+3)/2
(define a:merge-two-fractions?
  (function*
    [ ((BinOp op1 n1 n2) (BinOp op2 m1 m2))
     #:if (and (eq? op1 op/) (eq? op2 op/) (eq? n2 m2))
     #t]
    [_ #f]))

(define merge-two-fractions
  (phi* ((BinOp op1 n1 n2) (BinOp op2 m1 m2))
      (BinOp op/ (BinOp op+ n1 m1) n2)))
      
  
; H. Evaluates binary operations, except for division with one exception:
; G. A number over itself is 1 ⇒ 5/5 =1

(define a:binop-eval?
  (function
    [(BinOp op (Number n1) (Number n2))
     #:if (not (eq? op op/))
     #t]
    [(BinOp op (Number n1) (Number n2))
     #:if (eq? n1 n2)
     #t]
    [_ #f]))

(define a:binop-eval
  (phi (BinOp op (Number n1) (Number n2))
       (Number (compute-bin-op op n1 n2))))

; ==============================
; ========   Tactics ===========
; ==============================

;;; (define MAX-SKIP 30)

;;; (define (t:consecutive-swaps f)
;;;   (let* ([term (Fact-term f)]
;;;          [l (SortingList-elems term)])
;;;     (map
;;;      (lambda (i)
;;;        (fact (sd:swap term i)
;;;              (FactProof sd:swap (list (FactId (Fact-id f)) i))))
;;;      (range (- (length l) 2)))))

;;; (define (t:reverse f)
;;;   (let ([term (Fact-term f)])
;;;     (list (fact (sd:reverse term)
;;;                 (FactProof sd:reverse (list (FactId (Fact-id f))))))))

;;; ; Domain function: given a node, lists all child nodes.
;;; (define (d:sorting facts)
;;;   (append
;;;    (t:consecutive-swaps (last facts))
;;;    (t:reverse (last facts))))



; Generates a random fraction problem 
(define (generate-fraction-problem [max-number-terms 3])
  (let* ([number-of-terms (random 1 (+ 1 max-number-terms))]
         [fractions (map (lambda (_) (generate-fraction))(range number-of-terms))])
 (Problem
      (list (assumption (FractionExpression (list-to-sum fractions))))
      (list (Number))) ; want it to be Number ||  (Number / Number)
    ))

; Convert a list of single elements into a sum through BinOp 
(define (list-to-sum fractions)
    (if (eq? (length fractions) 1)
        fractions
        (BinOp op+ (car fractions) (list-to-sum (cdr fractions))) ))

; define the list of prime factors allowed in the generated exercises
(define primes '(1 2 3 5 7))

; Generate a single fraction
(define (generate-fraction [max-number-primes-factors 7])
  (let* (
         [len-primes-numerator (random 1 max-number-primes-factors)]
         [len-primes-denominator (random 1 max-number-primes-factors)]
         [list-primes-numerator (map (lambda (_) (list-ref primes (random 0 (length primes)))) (range len-primes-numerator))]
         [list-primes-denominator (map (lambda (_) (list-ref primes (random 0 (length primes))))(range len-primes-denominator))]
         [numerator (foldl * 1 list-primes-numerator)]
         [denominator (foldl * 1 list-primes-denominator)])
        
        (BinOp op/ numerator denominator)))

(define (is-fraction-simplified? f g)
  (let ([x (FractionExpression-elems (Fact-term f))])
    #f))

(provide
 generate-fraction-problem
 generate-fraction
 )
 
