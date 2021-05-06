import React, { useState } from 'react'
import { useRouter } from 'next/router'

const Instructions = () => {
  const [checked, setChecked] = useState(false);

  const [q1, setQ1] = useState(false);
  const [q2, setQ2] = useState(false);
  const [q3, setQ3] = useState(false);
  const [q4, setQ4] = useState(false);
  const [q5, setQ5] = useState(false);

  const router = useRouter();

  const next = () => {
    if (!(!q1 && q2 && !q3 && q4 && !q5)) {
      alert('Please carefully read the instructions to answer the questions correctly!');
      return;
    }
    router.push('/td-instructions');
  };

  return (
    <div className="content">
      <h1>Human or Machine: Step-by-Step Solutions for Equation Problems</h1>
      <p>
        In this experiment, you will be shown [20] equation problems.
        For each equation problem, you will see 4 different step-by-step solutions, in which two are written
        by human, and two by machines. All of the solutions are valid and mistakes-free.
        Your task is to rank the 4 solutions based on your confidence in which one is written by human.

        [alternatively: your task is to decide which two of the four solutions are written by human].

        The experiment should take around [25] minutes.
    </p>

    <p>
      By following the instructions, you are participating in a study being performed by
      cognitive scientists in the Stanford Department of Psychology.
      If you have questions about this research, please contact Gabriel Poesia at poesia@stanford.edu
      or Noah Goodman at ngoodman@stanford.edu.
    </p>

    <h2>Consent</h2>

    <ul>
      <li>You must be at least 18 years old to participate.</li>
      <li>Your participation in this research is voluntary.</li>
      <li>You may decline to answer any or all of the following questions.</li>
      <li>You may decline further participation, at any time, without adverse consequences.</li>
      <li>Your anonymity is assured; the researchers who have requested your participation will not receive any personally identifying information.</li>
    </ul>

    <p>
      <input type="checkbox" value={checked} onChange={() => setChecked(!checked)} />
      <span>I have read and agree with the above.</span>
    </p>

    <h2>Checking your understanding</h2>

    <p>To start the experiment, please check only the options that are true:</p>

    <ul className="invisible-list">
      <li><input type="checkbox" value={q1} onChange={() => setQ1(!q1)} /><span>The experiment involves solving equations and finding the right answers.</span></li>
      <li><input type="checkbox" value={q2} onChange={() => setQ2(!q2)} /><span>The exercises involves ranking step-by-step solutions based on your belief in which one is written by human.</span></li>
      <li><input type="checkbox" value={q3} onChange={() => setQ3(!q3)} /><span>Some of the solutions contain algebra mistakes.</span></li>
      <li><input type="checkbox" value={q4} onChange={() => setQ4(!q4)} /><span>In "AABAC", swapping the order of the last two characters gives "AABCA"</span></li>
        <li><input type="checkbox" value={q5} onChange={() => setQ5(!q5)} /><span>In "ABZBA", erasing the character "Z" results in "ABB".</span></li>
    </ul>

    <button disabled={!checked} onClick={next}>Continue</button>

    </div>
  );
};
export default Instructions;
