; Script to generate problems and solutions offline.

#lang racket

(require "generation.rkt")

(generate-problems-job (processor-count) "problem-database.json")
