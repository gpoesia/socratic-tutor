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

; Add to the current number, moving it to the previous number slot.
(define sd:swap
  (phi* ((SortingList l) i)
        (let-values ([(l-prefix l-tail) (split-at l i)])
          (SortingList (append l-prefix
                               (list (cadr l-tail) (car l-tail))
                               (drop l-tail 2))))))

(define sd:reverse
  (phi (SortingList l) (SortingList (reverse l))))
; ==============================
; ========   Tactics ===========
; ==============================

(define MAX-SKIP 30)

(define (t:consecutive-swaps f)
  (let* ([term (Fact-term f)]
         [l (SortingList-elems term)])
    (map
     (lambda (i)
       (fact (sd:swap term i)
             (FactProof sd:swap (list (FactId (Fact-id f)) i))))
     (range (- (length l) 2)))))

(define (t:reverse f)
  (let ([term (Fact-term f)])
    (list (fact (sd:reverse term)
                (FactProof sd:reverse (list (FactId (Fact-id f))))))))

; Domain function: given a node, lists all child nodes.
(define (d:sorting facts)
  (append
   (t:consecutive-swaps (last facts))
   (t:reverse (last facts))))

; Generates a random ternary addition problem by generating random digits.
(define (generate-sorting-problem [max-difficulty 10])
  (let* ([len (random 2 max-difficulty)]
         [l (shuffle (range 1 (+ 1 len)))])
    (Problem
     (list (assumption (SortingList l)))
     (list (AnyNumber)))))

(define (is-sorting-list-sorted? f g)
  (let ([l (SortingList-elems (Fact-term f))])
    (equal? l (sort l <))))

(provide
 sd:reverse
 sd:swap
 d:sorting
 generate-sorting-problem
 is-sorting-list-sorted?)
