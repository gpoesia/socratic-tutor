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
(require "debug.rkt")

(struct Domain (name generator verifier step) #:transparent)

(struct Problem (initial-facts goals) #:transparent)

(define (format-problem p)
  (format "Given ~a solve ~a"
          (string-join (map format-fact (Problem-initial-facts p)) ", ")
          (string-join (map format-term (Problem-goals p)) ", ")))

(struct SolverResult (facts
                      met-goals
                      unmet-goals
                      contradiction) #:transparent)

; A state in the tree search.
(struct MCTSNode (facts [value #:mutable] [is-leaf? #:mutable]) #:transparent)

(struct MCTSResult (nodes terminal) #:transparent)

; Given a SolverResult, returns whether the problem was solved successfully.
(define (problem-solved? sr) (empty? (SolverResult-unmet-goals sr)))

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
         [(new-facts) (append renewed-facts (prune next-old-facts dedup-new-facts))]
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

; Searches for a solution using a simple Beam Search.
(define (find-solution-smc-loop
         nodes
         goals
         domain
         value-function
         n-samples
         max-depth)
  ; If timed out, return.
  (if (= 0 max-depth)
    (MCTSResult nodes #f)
    ; Find leaf with max value.
    (let* ([all-leaves (filter MCTSNode-is-leaf? nodes)]
           [leaves (take (sort (shuffle all-leaves)
                               (lambda (n1 n2) (> (MCTSNode-value n1) (MCTSNode-value n2))))
                         (min n-samples (length all-leaves)))]
           [proposals (apply append
                             (map (lambda (n)
                                    (map (lambda (f)
                                           (MCTSNode (append (MCTSNode-facts n) (list f))
                                                      (MCTSNode-value n)
                                                      #t))
                                         ((Domain-step domain) (MCTSNode-facts n))))
                                  leaves))]
           ; For each child node, check whether it solves all goals.
           [terminal-nodes (filter (lambda (node) (solves-problem? goals
                                                                   (MCTSNode-facts node)
                                                                   (Domain-verifier domain)))
                                   proposals)]
           ; Compute value estimates using value function.
           [proposal-values (if (empty? terminal-nodes) (value-function proposals) (list))])
      ; Expanded nodes are not leaves anymore; update it.
      (for-each (lambda (node) (set-MCTSNode-is-leaf?! node #f)) nodes)
      (if (not (empty? terminal-nodes))
        ; Found a solution!
        (MCTSResult (append nodes proposals) (car terminal-nodes))
        ; Otherwise, recurse.
        (begin
          ; Update computed values.
          (for-each (lambda (node value) (set-MCTSNode-value! node (+ (MCTSNode-value node)
                                                                      (log value))))
                    proposals proposal-values)
          (find-solution-smc-loop (append nodes proposals)
                                  goals
                                  domain
                                  value-function
                                  n-samples
                                  (- max-depth 1)))))))

(define (inverse-term-size-value-function nodes)
  (map (lambda (node) (/ 1 (term-size (Fact-term (last (MCTSNode-facts node))))))
       nodes))

(define (random-value-function nodes)
  (map (lambda (node) (random)) nodes))

(define (solve-problem-smc problem domain value-function n-samples max-depth)
  (let ([initial-node (MCTSNode (Problem-initial-facts problem) 0.0 #t)])
    (if (solves-problem? (Problem-goals problem) (MCTSNode-facts initial-node) (Domain-verifier domain))
        (MCTSResult (list initial-node) initial-node)
        (find-solution-smc-loop
         (list initial-node)
         (Problem-goals problem)
         domain
         value-function
         n-samples
         max-depth))))

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

; Returns whether the list of facts given solves all goals.
(define (solves-problem? goals facts [check fact-solves-goal?])
  (let-values ([(_ unmet) (match-goals empty goals facts check)])
    (empty? unmet)))

; Checks whether all goals in unmet-goals match any of the facts.
; Returns a pair (met-goals . unmet-goals).
(define (match-goals met-goals unmet-goals facts [check fact-solves-goal?])
  (if (empty? unmet-goals)
    (values met-goals unmet-goals)
    (let*-values ([(g rest) (values (car unmet-goals) (cdr unmet-goals))]
                  [(met-goals-r unmet-goals-r) (match-goals met-goals rest facts check)])
      ; If g matches any of the facts, add it to met goals. Otherwise,
      ; to unmet goals.
      (if (ormap (lambda (f) (check f g)) facts)
        (values (cons g met-goals-r) unmet-goals-r)
        (values met-goals-r (cons g unmet-goals-r))))))

; Returns which of the facts in the SolverResult sr matches the goal g.
(define (goal-solution g sr)
  (findf (lambda (f) (fact-solves-goal? f g)) (SolverResult-facts sr)))

; Don't prune: keep all facts.
(define (prune:keep-all all-facts facts) facts)

; Returns a pruner function that sorts facts by term size and gets the k
; smallest, with a random tie breaking.
(define (prune:keep-smallest-k k)
  (lambda (all-facts facts)
    (take (sort-facts-by-size facts) (min k (length facts)))))

; Returns a pruner function that samples k facts uniformly.
(define (prune:keep-random-k k)
  (lambda (all-facts facts)
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

(define (trace-solver-steps all-facts facts)
  (let* ([fact-trace-ids (trace-facts all-facts facts)]
         [relevant-facts (filter (lambda (sf) (member (Fact-id sf) fact-trace-ids))
                                 (remove-duplicates (append facts all-facts)))])
    (renumber relevant-facts)))

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
  Domain Domain-name Domain-generator Domain-verifier Domain-step
  Problem? Problem Problem-initial-facts Problem-goals format-problem
  MCTSNode MCTSResult MCTSNode-facts MCTSNode-value MCTSResult-terminal MCTSNode-is-leaf? MCTSResult-nodes
  SolverResult SolverResult? problem-solved?
  SolverResult-facts
  SolverResult-met-goals
  SolverResult-unmet-goals
  SolverResult-contradiction
  find-solution
  solve-problem
  solve-problem-smc
  solves-problem?
  inverse-term-size-value-function
  random-value-function
  goal-solution
  get-step-by-step-solution
  get-step-by-step-contradiction
  trace-facts
  trace-solver-steps
  renumber
  prune:keep-all
  prune:keep-smallest-k
  prune:keep-random-k)
