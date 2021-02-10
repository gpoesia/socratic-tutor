import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { apiRequest } from '../lib/api';
import useStore from '../lib/state';
import { sprintf } from 'sprintf-js';
import _ from 'lodash';

const TIMEOUT = 180;

const Assessment = () => {
  const router = useRouter();
  const [{ problems, answers }, setProblemsAndAnswers] = useState({});
  const [begin, setBegin] = useState(new Date());
  const [now, setNow] = useState(new Date());
  const id = useStore(state => state.id);

  const testIndex = router.query.stage === "pre" ? 1 : 2;

  useEffect(async () => {
    if (problems !== undefined) {
      return;
    }

    const ps = await apiRequest('test-problems', { id });
    console.log('Got problems:', ps);
    setProblemsAndAnswers({ problems: ps,
                            answers: _.fromPairs(ps.map(p => [p.id, ""])) });
  });

  useEffect(() => {
    const t = setTimeout(() => setNow(new Date()), 1000);
    return () => clearTimeout(t);
  });

  const elapsedSeconds = (now - begin) / 1000;
  const remainingSeconds = Math.floor(Math.max(TIMEOUT - elapsedSeconds, 0));
  const remainingTime = sprintf("%02d:%02d",
                                Math.floor(remainingSeconds / 60),
                                remainingSeconds % 60);

  const submit = async () => {
    if (_.size(answers) == 0 || _.size(answers) < _.size(problems)) {
      return alert("Please provide an answer to all the problems.");
    }

    const invalid = _.find(answers, (value) => isNaN(parseInt(value))) !== undefined;
    if (invalid) {
      return alert("Please answer each equation with a number.");
    }
    console.log('Answers:', answers);
    const userAnswers = _.map(answers, (value, key) => ({ id: key, answer: parseInt(value) }));
    console.log('User answers:', userAnswers);
    const response = await apiRequest('save-answers',
                                      {
                                        id,
                                        stage: "assessment-" + router.query.stage,
                                        answers: userAnswers,
                                      });

    console.log('Response:', response);

    if (!response.error) {
      if (router.query.stage == "pre") {
        router.push('/exercises');
      } else {
        router.push('/end');
      }
    }
  };

  return (
    <div className="content">
      <h1>Assessment {testIndex}/2</h1>
      <h4>Remaining time: { remainingTime }</h4>

      <p>
        In this assessment, solve the equations below to the best of your knowledge.
        Type your solution to each equation in the boxes.
        All solutions will be integer numbers.
        You are allowed to use pencil, paper and a calculator, if needed.
      </p>

      <div>
        { (problems || []).map((p, i) => (
          <div key={i} className="problem">
            <p>{ p.problem }</p>
            <span>
              { p.variable } = &nbsp;
              <input
                type="text" size="1" value={answers[p.id]}
                onChange={e => setProblemsAndAnswers(
                                   {
                                     problems,
                                     answers: {
                                        ...answers,
                                       [p.id]: e.target.value,
                                     },
                                   })} />
            </span>
          </div>
        )) }

        <button onClick={submit}>Submit answers</button>
      </div>
    </div>
  );
};
export default Assessment;
