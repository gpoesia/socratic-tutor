; Implementation of a Hilbert-style deductive system for solving
; math problems.

#lang algebraic/racket/base

(require data/gvector)
(require racket/set)
(require racket/string)
(require racket/list)
(require racket/function)

(require "terms.rkt")
(require "facts.rkt")
(require "tactics.rkt")
(require "debug.rkt")

(struct Problem (initial-facts goals) #:transparent)

(define (format-problem p)
  (format "Given ~a solve ~a"
          (string-join (map format-fact (Problem-initial-facts p)) ", ")
          (string-join (map format-term (Problem-goals p)) ", ")))

(struct SolverResult (facts
                      met-goals
                      unmet-goals
                      contradiction) #:transparent)

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
             (string-join (map format-fact last-facts) ", "))
  ; Base cases: either solved the problem, timed out or found a contradiction.
  (let ([contradiction (find-contradiction last-facts)])
    (if (or (= 0 depth) (empty? unmet-goals) contradiction)
      (SolverResult (append old-facts last-facts)
                    met-goals unmet-goals contradiction)
      (let*-values
        ([(new-facts-unfiltered kept-old-facts new-strategy-state)
            (strategy old-facts last-facts unmet-goals strategy-state)]
         [(next-old-facts-ids) (trace-facts
                                 (append old-facts last-facts)
                                 (append kept-old-facts new-facts-unfiltered))]
         [(next-old-facts) (filter (lambda (f)
                                     (and
                                       (member (Fact-id f) next-old-facts-ids)
                                       (not (member f new-facts-unfiltered))))
                                   (append old-facts last-facts))]
         [(renewed-facts) (filter (lambda (f)
                                    (or (member f old-facts)
                                        (member f last-facts)))
                                  new-facts-unfiltered)]
         [(dedup-new-facts) (remove* (append next-old-facts renewed-facts)
                                     (remove-duplicates
                                       new-facts-unfiltered
                                       #:key Fact-term)
                                     fact-terms-equal?)]
         [(new-facts) (append renewed-facts (prune dedup-new-facts))]
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
          (- depth 1))))))

(define is-contradiction?
  (function
    [(Predicate 'Eq ((Number a) (Number b)))
     #:if (not (= a b))
     #t]
    [_ #f]
    ))

(define (find-contradiction facts)
  (findf (lambda (f) (is-contradiction? (Fact-term f))) facts))

(define (find-solution goals initial-facts strategy prune depth)
  (find-solution-loop empty initial-facts empty goals #f strategy prune depth))

(define (solve-problem problem strategy prune depth)
  (find-solution (Problem-goals problem)
                 (Problem-initial-facts problem)
                 strategy prune depth))

; Checks whether all goals in unmet-goals match any of the facts.
; Returns a pair (met-goals . unmet-goals).
(define (match-goals met-goals unmet-goals facts)
  (if (empty? unmet-goals)
    (values met-goals unmet-goals)
    (let*-values ([(g rest) (values (car unmet-goals) (cdr unmet-goals))]
                  [(met-goals-r unmet-goals-r) (match-goals met-goals rest facts)])
      ; If g matches any of the facts, add it to met goals. Otherwise,
      ; to unmet goals.
      (if (ormap (lambda (f) (fact-solves-goal? f g)) facts)
        (values (cons g met-goals-r) unmet-goals-r)
        (values met-goals-r (cons g unmet-goals-r))))))

; Returns which of the facts in the SolverResult sr matches the goal g.
(define (goal-solution g sr)
  (findf (lambda (f) (fact-solves-goal? f g)) (SolverResult-facts sr)))

; Don't prune: keep all facts.
(define prune:keep-all identity)

; Returns a pruner function that sorts facts by term size and gets the k
; smallest, with a random tie breaking.
(define (prune:keep-smallest-k k)
  (lambda (facts) (smallest-k-facts k facts)))

; Returns a pruner function that samples k facts uniformly.
(define (prune:keep-random-k k)
  (lambda (facts)
    (take (shuffle facts) (min k (length facts)))))

; Returns a step-by-step solution from a SolverResult
; only involving relevant facts.
(define (get-step-by-step-solution sr)
  (let* ([solutions (map (lambda (g) (goal-solution g sr))
                         (SolverResult-met-goals sr))]
         [all-solution-steps (trace-facts (SolverResult-facts sr)
                                          solutions)]
         [relevant-facts (filter (lambda (f) (member (Fact-id f)
                                                     all-solution-steps))
                                 (SolverResult-facts sr))])
    (renumber relevant-facts)))

; Topologically sort facts by their proof dependencies.
; Worst-case O(N^3), but always called on lists containing just solution
; steps, which are very small.
(define (facts-toposort facts [included empty])
  (let* ([next (filter
                 (lambda (f)
                   (and (not (member (Fact-id f) included))
                        (andmap (lambda (d) (member d included))
                                (fact-dependencies f))))
                 facts)])
    (if (empty? next)
      empty
      (append next (facts-toposort facts (append included (map Fact-id next)))))))

; Returns a step-by-step derivation of a contradiction from a
; SolverResult, in case the solver found one.
(define (get-step-by-step-contradiction sr)
  (if (SolverResult-contradiction sr)
    (let* ([all-steps (trace-facts
                        (SolverResult-facts sr)
                        (list (SolverResult-contradiction sr)))]
           [relevant-facts (filter (lambda (f) (member (Fact-id f)
                                                       all-steps))
                                   (SolverResult-facts sr))])
      (renumber relevant-facts))
    #f))

; Returns a list of the ids of all facts that appear in the proof
; of any f in `facts` (including themselves).
(define (trace-facts all-facts facts)
  (if (empty? facts)
    empty
    (apply append
      (map
        (lambda (f)
          (let* ([proof-arguments (FactProof-parameters (Fact-proof f))]
                 [dependencies (fact-dependencies f)])
            (append (list (Fact-id f))
                    dependencies
                    (trace-facts all-facts
                      (filter (lambda (f) (member (Fact-id f) dependencies))
                              all-facts))))
          )
        facts))))

; Given a list of facts, changes their IDs to be sequentially assigned
; integers. Useful for human-friendly presentation.
(define (renumber facts)
  (for/fold ([new-id (hash)]
             [new-facts (list)]
             #:result (reverse new-facts))
            ([f (in-list (facts-toposort facts))]
             [i (in-range 1 (+ 1 (length facts)))])
    (let ([proof (Fact-proof f)])
      (values
        (hash-set new-id (Fact-id f) i)
        (cons
          (Fact i ; New ID.
                (Fact-term f) ; Same term.
                (FactProof
                  (FactProof-axiom proof) ; Same axiom.
                  (map (lambda (p) ; Rewrite proof parameters.
                         (if (FactId? p)
                           (FactId (hash-ref new-id (FactId-id p)))
                           p))
                       (FactProof-parameters proof))))
          new-facts)))))

(provide
  Problem? Problem Problem-initial-facts Problem-goals format-problem
  SolverResult SolverResult?
  SolverResult-facts
  SolverResult-met-goals
  SolverResult-unmet-goals
  SolverResult-contradiction
  find-solution
  solve-problem
  goal-solution
  get-step-by-step-solution
  get-step-by-step-contradiction
  trace-facts
  renumber
  prune:keep-all
  prune:keep-smallest-k
  prune:keep-random-k)
