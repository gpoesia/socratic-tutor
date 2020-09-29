; Debug macros.
#lang racket

(require date)
(require syntax/location)
(require syntax/srcloc)

; Change to true to enable debugging
(define debug? #f)
; Only print messages that contain this pattern.
(define debug-pattern #rx"")

(define-syntax-rule (log-debug message args ...)
  (if debug?
    (let ([loc (quote-srcloc)])
      (if (regexp-match debug-pattern message)
        (printf (string-append "[~a:~a@~a]: " message)
                (source-location-source loc)
                (source-location-line loc)
                (current-date-string-iso-8601 #t)
                args ...)
        (void)))
    (void)))

(provide log-debug)
