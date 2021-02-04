import React, { useState } from 'react';
// import GuessingGame from './guessing-game.js';
import dynamic from 'next/dynamic';

const DynGuessingGame = dynamic(() => import('../lib/components/guessing-game.js'),
                                { ssr: false });

const policies = [
  { display: "[Choose]" },
  { display: "Random",
    policy: "random" },
  { display: "Curriculum",
    policy: "curriculum" },
  { display: "Personalized Curriculum",
    policy: "personalized_curriculum" },
];

function ProblemSelection() {
  const [chosenPolicy, setChosenPolicy] = useState(null);

  if (chosenPolicy) {
    return (
      <DynGuessingGame policy={chosenPolicy} />
    );
  } else {
    return (
      <div className='container'>
        <h1>Pick a problem selection policy</h1>

        <select onChange={e => setChosenPolicy(policies[e.target.value].policy)}>
          {
            policies.map((p, i) => <option key={i} value={i}>
                                     {p.display}
                                   </option>)
          }
        </select>
      </div>
    );
  }
}

export default function App() {
  return (
    <ProblemSelection />
  );
}
