; Implementation of a Hilbert-style deductive system for solving
; math problems.

#lang racket

(require data/gvector)
(require racket/set)

(require "terms.rkt")
(require "tactics.rkt")
(require "debug.rkt")
(require "strategy.rkt")

(struct SolverResult (facts met-goals unmet-goals) #:transparent)

; Searches for a solution for the given set of goals.
; old-facts - list of facts known as of the last iteration
; last-facts - list of facts discovered in the last iteration
; met-goals - list of goals that match some known fact
; unmet-goals - list of goals that were still not met by
;               any known fact.
; strategy-state - used by the strategy
; strategy - function that, given facts, unmet-goals and strategy-state
;            returns a tactic to apply plus modified state.
; prune - function that, given a list of newly produced facts,
;         will (possibly) reduce the list, pruning facts that will not
;         be needed.
; depth - search depth.
; Returns a SolverResult value.
(define (find-solution-loop
          old-facts
          last-facts
          met-goals
          unmet-goals
          strategy-state
          strategy 
          prune
          depth)
  (log-debug "find-solution-loop depth ~a, ~a new facts: ~a\n"
             depth
             (length last-facts)
             (string-join (map format-term last-facts) ", "))
  ; Base cases: either solved the problem or timed out.
  (if (or (= 0 depth) (empty? unmet-goals))
    (SolverResult (append old-facts last-facts) met-goals unmet-goals)
    (let*-values
      ([(next-tactic new-strategy-state)
          (strategy old-facts last-facts unmet-goals strategy-state)]
       [(new-facts-unfiltered)
          (next-tactic met-goals unmet-goals old-facts last-facts)]
       [(next-old-facts) (append old-facts last-facts)]
       [(new-facts) (prune (remove* next-old-facts (remove-duplicates new-facts-unfiltered)))]
       [(new-met-goals new-unmet-goals) 
          (match-goals met-goals unmet-goals new-facts)])
      (find-solution-loop
        next-old-facts
        new-facts
        new-met-goals
        new-unmet-goals
        new-strategy-state
        strategy
        prune
        (- depth 1)))))

(define (find-solution goals initial-facts strategy prune depth)
  (find-solution-loop empty initial-facts empty goals #f strategy prune depth))

; Checks whether all goals in unmet-goals match any of the facts.
; Returns a pair (met-goals . unmet-goals).
(define (match-goals met-goals unmet-goals facts)
  (if (empty? unmet-goals)
    (values met-goals unmet-goals)
    (let*-values ([(g rest) (values (car unmet-goals) (cdr unmet-goals))]
                  [(met-goals-r unmet-goals-r) (match-goals met-goals rest facts)])
      ; If g matches any of the facts, add it to met goals. Otherwise,
      ; to unmet goals.
      (if (ormap (lambda (f) (goal-matches? g f)) facts)
        (values (cons g met-goals-r) unmet-goals-r)
        (values met-goals-r (cons g unmet-goals-r))))))

; Returns which of the facts in the SolverResult sr matches the goal g.
(define (goal-solution g sr)
  (findf (lambda (f) (goal-matches? g f)) (SolverResult-facts sr)))

; Don't prune: keep all facts.
(define prune:keep-all identity)

; Returns a pruner function that sorts facts by term size and gets the k
; smallest, with a random tie breaking.
(define (prune:keep-smallest-k k)
  (lambda (facts)
    (take
      (sort (shuffle facts)
            (lambda (a b) (< (term-size a) (term-size b))))
      (min k (length facts)))))

; Returns a pruner function that samples k facts uniformly.
(define (prune:keep-random-k k)
  (lambda (facts)
    (take (shuffle facts) (min k (length facts)))))

(provide
  SolverResult
  SolverResult-facts
  SolverResult-met-goals
  SolverResult-unmet-goals
  find-solution
  goal-solution
  prune:keep-all
  prune:keep-smallest-k
  prune:keep-random-k)
