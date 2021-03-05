#lang racket

(require net/url
         (prefix-in sequencer: web-server/dispatchers/dispatch-sequencer)
         web-server/dispatch
         web-server/servlet
         web-server/servlet-env
         web-server/http
         web-server/servlet-dispatch
         web-server/web-server)
(require json)

(require "terms.rkt")
(require "facts.rkt")
(require "solver.rkt")
(require "term-parser.rkt")
(require "domains.rkt")
(require "questions.rkt")
(require "serialize.rkt")

(define (make-post-endpoint f)
  (lambda (req)
    (let* ([data-raw (request-post-data/raw req)]
           [params (bytes->jsexpr data-raw)])
      (response/output
       (lambda (out) (display (to-json-string (f params)) out))))))

(define api:generate
  (make-post-endpoint
    (lambda (params)
      (let* ([domain-name (hash-ref params 'domain)]
             [domain (get-domain-by-name domain-name)]
             [seed (if (hash-has-key? params 'seed)
                       (random-seed (hash-ref params 'seed))
                       (void))]
             [generator (Domain-generator domain)]
             [problem (generator)])
        (hash
         'state (map format-fact (Problem-initial-facts problem))
         'goals (map format-term (Problem-goals problem)))))))

(define api:step
  (make-post-endpoint
    (lambda (params)
      (let* ([domain-name (hash-ref params 'domain)]
             [domain (get-domain-by-name domain-name)]
             [states-str (hash-ref params 'states)]
             [states (map (lambda (s)
                            (map (lambda (t) (assumption (parse-term t))) s))
                            states-str)]
             [goals-str (hash-ref params 'goals)]
             [goals (map (lambda (g) (map parse-term g)) goals-str)]
             [verifier (Domain-verifier domain)]
             [step-fn (Domain-step domain)])
        (map (lambda (state goals)
               (if (solves-problem? goals state verifier)
                   (hash 'success #t 'actions empty)
                   (let ([next-facts (step-fn state)])
                     (hash
                      'success #f
                      'actions (map (lambda (f)
                                      (let ([desc (generate-formal-step-description (Fact-proof f) state)])
                                        (hash 'id (Fact-id f)
                                              'state (format-fact f)
                                              'action desc)))
                                    next-facts)))))
             states goals)))))

(define api:error
  (lambda (req)
    (response/jsexpr "Error: route not found (maybe check that you made a POST request?)")))

(define-values (routes reverse-uri)
  (dispatch-rules
   [("generate") #:method POST api:generate]
   [("step") #:method POST api:step]
   [else api:error]
   ))

(define host (make-parameter "127.0.0.1"))
(define port (make-parameter 9898))

(command-line
  #:program "environment-server"
  #:once-each
  [("-H" "--host") H
   "Host address to bind to."
   (host H)]
  [("-p" "--port") p
   "Port to serve on."
   (port (string->number p))])

(define stop
  (begin
    (printf "Running environment server on ~a:~a\n" (host) (port))
    (serve
      #:dispatch (sequencer:make (dispatch/servlet routes))
      #:listen-ip (host)
      #:port (port))))

(with-handlers ([exn:break? (lambda (e) (stop))])
               (sync/enable-break never-evt))
