import React, { useState } from 'react';
import TutorDialogue from './tutor-dialogue.js';

const makeProblem = (facts, goals) => ({ facts, goals });

const problems = [
  { display: "" },
  { display: "2x = 4",
    problem: makeProblem(["2x = 4"], ["x = ?"]) },
  { display: "2x + 1 = 5",
    problem: makeProblem(["2x + 1 = 5"], ["x = ?"]) },
  { display: "x = 2 + 2, y = x + 1",
    problem: makeProblem(["x = 2 + 2", "y = x + 1"], ["x = ?", "y = ?"]) },
  { display: "x = y/2, y = x - 1",
    problem: makeProblem(["x = y/2", "y = x - 1"], ["x = ?", "y = ?"]) },
  { display: "x = (25*16) + 3*(14 + 4 - 9)",
    problem: makeProblem(["x = (25*16) + 3*(14 + 4 - 9)"], ["x = ?"]) },
];

export default function ProblemSelection() {
  const [chosenProblem, setChosenProblem] = useState(null);

  if (chosenProblem) {
    return (
      <TutorDialogue problem={chosenProblem} />
    );
  } else {
    return (
      <div className='container'>
        <h1>Pick a problem</h1>

        <select onChange={e => setChosenProblem(problems[e.target.value].problem)}>
          {
            problems.map((p, i) => <option key={i} value={i}>
                                     {p.display}
                                   </option>)
          }
        </select>
      </div>
    );
  }
}
