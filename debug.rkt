; Debug macros.
#lang racket

(require date)
(require syntax/location)
(require syntax/srcloc)

; Change to true to enable debugging
(define debug? #f)

(define-syntax-rule (log-debug message args ...)
  (if debug?
    (let ([loc (quote-srcloc)])
      (printf (string-append "[~a:~a@~a]: " message)
              (source-location-source loc)
              (source-location-line loc)
              (current-date-string-iso-8601 #t)
              args ...))
    (void)))

(provide log-debug)
