#lang racket

(require "terms.rkt")
(require "learn.rkt")
(require "equations.rkt")
(require "domains.rkt")
(require "questions.rkt")
(require "facts.rkt")
(require "solver.rkt")
(require "term-parser.rkt")
(require "value-function.rkt")

(define depth (make-parameter 5))
(define domain (make-parameter "equations"))
(define use-value-function (make-parameter #f))
(define beam-width (make-parameter 10))
(define server (make-parameter "http://127.0.0.1:9911/"))

(command-line
  #:program "solver-repl"
  #:once-each
  [("-V" "--value-function")
   "Use neural value function server."
   (use-value-function #t)]
  [("-S" "--server") s
   "URL of the server to access the neural value function."
   (server s)]
  [("-d" "--depth") d
   "Max search depth."
   (depth (string->number d))]
  [("-D" "--domain") D
   "Name of the domain.."
   (domain D)]
  [("-b" "--beam") b
   "Beam width for beam search."
   (beam-width (string->number b))])

(define (solve-and-print-solution input-param)
  (let* ([term (parse-term input-param)]
         [problem (Problem (list (assumption term)) (list (parse-term "x = ?")))]
         [result (solve-problem-smc
                  problem
                  (get-domain-by-name (domain))
                  (if (use-value-function)
                      (make-neural-value-function (server))
                      inverse-term-size-value-function)
                  (beam-width)
                  (depth))]
         [solution (MCTSResult-terminal result)])
    (printf "~a\n"
            (if solution
                (string-join (map (lambda (f)
                                    (format "~a [~a]"
                                            (format-fact f)
                                            (generate-step-description
                                             (Fact-proof f) (MCTSNode-facts solution))))
                                  (MCTSNode-facts solution)) "\n")
                (begin
                  (for-each (lambda (node) (printf "~a :: ~a\n"
                                                   (string-join
                                                    (format-step-by-step-terms (MCTSNode-facts node))
                                                    "=>")
                                                   (MCTSNode-value node)))
                            (MCTSResult-nodes result))
                  "<no solution found>")))))

(define (print-next-step input-param)
  (let* ([facts (list (assumption (parse-term input-param)))]
         [domain (get-domain-by-name (domain))]
         [next-facts ((Domain-step domain) facts)]
         [next-nodes (map (lambda (f) (MCTSNode (append facts (list f)) 0 #f)) next-facts)]
         [value-fn (if (use-value-function)
                       (make-neural-value-function (server))
                       inverse-term-size-value-function)]
         [values (value-fn next-nodes)]
         [nodes-with-value (map cons next-nodes values)]
         [sorted-nodes (sort nodes-with-value (lambda (n1 n2) (< (cdr n1) (cdr n2))))])
    (for-each (lambda (n) (printf "~a :: ~a\n"
                                  (cdr n)
                                  (format-term (Fact-term (last (MCTSNode-facts (car n)))))))
              sorted-nodes)))

(define (main)
  (with-handlers ([exn:break? (lambda (e) (printf "Interrupted.\n") (main))])
    (printf ">> ")
    (flush-output)
    (let* ([input (read-line (current-input-port) 'any)]
           [break? (if (eof-object? input) (raise-user-error "Exiting...") #f)]
           [solve-command? (string-prefix? input "s ")]
           [step-command? (string-prefix? input "n ")]
           [quit? (string-prefix? input "q")]
           [input-param (list->string (drop (string->list input) (min (string-length input) 2)))])
      (if (not (or quit? solve-command? step-command?))
       (begin (printf "Syntax: s <problem> or n <problem>\n") (main))
       (cond
         [quit? (void)]
         [solve-command? (solve-and-print-solution input-param) (main)]
         [step-command? (print-next-step input-param) (main)])))))

(main)
(provide main)
