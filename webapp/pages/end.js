import React, { useState } from 'react';

import { apiRequest } from '../lib/api';
import useStore from '../lib/state';

const End = () => {
  const id = useStore(state => state.id);
  const [age, setAge] = useState("18");
  const [education, setEducation] = useState("ms");
  const [experience, setExperience] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const submit = async () => {
    await apiRequest('save-survey', { id, survey: { age, education, experience } });
    setSubmitted(true);
  }

  return (
    <div className="content">
      <h1>Thank you!</h1>
      <p>
        This is the end of the experiment. Thank you for your participation.
        To finalize, we'd just like to know a little bit more about your experience.
      </p>

      <p>
        What is your age? <input type="text" size="1" value={age}
                                 onChange={e => setAge(e.target.value)} />
      </p>
      <p>
        What is the highest academic degree you have attained?
        <select onChange={e => setEducation(e.target.value)}>
          <option value="ms">Middle school</option>
          <option value="hs">High school</option>
          <option value="hs">College degree or above</option>
        </select>
      </p>
      <p>
        How did you like the experience?
        Do you think the generated exercises could be useful to teach algebra?
        Any feedback on how to improve the experience?
        <br/>
        <textarea rows="5"
                  cols="80"
                  value={experience}
                  onChange={e => setExperience(e.target.value)} />
      </p>
      {
        submitted
        ? <div>
            <p>
              To claim your reward on Mechanical Turk, copy the code below and
              paste it on the HIT survey:
            </p>
            <pre>{ id }</pre>
            <p>
              After copying the code, you may this tab. Have a good rest of your day!
            </p>
          </div>
        : <button onClick={submit}>Finish</button>
      }
    </div>
  );
};
export default End;
