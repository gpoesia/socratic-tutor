; Axioms and tactics used by the "ternary numbers domain" solver.
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

; Add consecutive digits with the same power.
(define td:add-consecutive
  (function*
   [(((TernaryDigit d1 p1) #:as td1) ((TernaryDigit d2 p2) #:as td2))
    #:if (= p1 p2)
    (list (TernaryDigit (quotient (+ d1 d2) 3) (+ 1 p1))
          (TernaryDigit (modulo (+ d1 d2) 3) p1))]
    [(td1 td2) #f]))

; Swap adjacent digits.
(define td:swap
  (function*
   [(((TernaryDigit d1 p1) #:as td1) ((TernaryDigit d2 p2) #:as td2))
;    #:if (< d1 d2)
    (list td2 td1)]
   [(td1 td2) #f]))

; Erase a zero digit.
(define td:erase-zero
  (function
   [(TernaryDigit 0 p) (list)]
   [t #f]))

; Returns a unique string representing each axiom.
(define (td:axiom->string a)
  (match a
    [(== 'Assumption) "Assumption"]
    [(== td:add-consecutive) "td:add-consecutive"]
    [(== td:swap) "td:swap"]
    [(== td:erase-zero) "td:erase-zero"]))

; Returns a unique string representing each axiom.
(define (string->axiom s)
  (match s
    [(== "Assumption") 'Assumption]
    [(== "td:add-consecutive") td:add-consecutive]
    [(== "td:swap") td:swap]
    [(== "td:erase-zero") td:erase-zero]))

; ==============================
; ========   Tactics ===========
; ==============================

; Tactics apply axioms to the new known facts, producing others.

(define-syntax-rule (consecutive-digits-tactic name transform)
  (define (name f)
    (let* ([term (Fact-term f)]
           [ds (TernaryNumber-digits term)])
      (filter identity
              (map (lambda (i)
                     (let*-values ([(before after) (split-at ds i)]
                                   [(d1) (car after)]
                                   [(d2) (cadr after)]
                                   [(tail) (drop after 2)]
                                   [(replacement) (transform d1 d2)])
                       (if replacement
                           (fact (TernaryNumber (append before replacement tail))
                                 (FactProof transform (list (FactId (Fact-id f)) i)))
                           #f)))
                   (range (- (length ds) 1)))))))

(define-syntax-rule (single-digit-tactic name transform)
  (define (name f)
    (let* ([term (Fact-term f)]
           [ds (TernaryNumber-digits term)])
      (filter identity
              (map (lambda (i)
                     (let*-values ([(before after) (split-at ds i)]
                                   [(d) (car after)]
                                   [(tail) (cdr after)]
                                   [(replacement) (transform d)])
                       (if replacement
                           (fact (TernaryNumber (append before replacement tail))
                                 (FactProof transform (list (FactId (Fact-id f)) i)))
                           #f)))
                   (range (length ds)))))))

(consecutive-digits-tactic tdt:swap td:swap)
(consecutive-digits-tactic tdt:add-consecutive td:add-consecutive)
(single-digit-tactic tdt:erase-zero td:erase-zero)

(define (combine-tactics tactics)
  (lambda (f)
    (apply append
           (map (lambda (t) (t f)) tactics))))

; Applies all tactics.
(define td:all (combine-tactics
                (list
                 tdt:swap
                 tdt:add-consecutive
                 tdt:erase-zero)))

; Domain function: given a node, lists all child nodes.
(define (d:ternary facts)
  (filter (lambda (f) (not (member f facts fact-terms-equal?)))
          (td:all (last facts))))

; Generates a random ternary addition problem by generating random digits.
(define (generate-ternary-addition-problem [max-difficulty 20] [max-power 5])
  (Problem
   (list (assumption (TernaryNumber (map (lambda (_) (TernaryDigit (random 0 3)
                                                                   (random 0 max-power)))
                                         (range (random 1 max-difficulty))))))
   (list (AnyNumber))))

(define (is-ternary-number-simplified? f g)
  (let* ([ternary-digits (TernaryNumber-digits (Fact-term f))]
         [powers (map TernaryDigit-power ternary-digits)]
         [digits (map TernaryDigit-digit ternary-digits)]
         [is-sorted? (equal? powers (sort powers <))]
         [powers-unique? (not (check-duplicates powers))]
         [has-zero? (member 0 digits)])
    (and is-sorted? powers-unique? (not has-zero?))))

(provide
 d:ternary
 generate-ternary-addition-problem
 is-ternary-number-simplified?

 td:add-consecutive
 td:swap
 td:erase-zero)
