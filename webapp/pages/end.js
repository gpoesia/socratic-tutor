import React, { useState } from 'react';

import { apiRequest } from '../lib/api';
import useStore from '../lib/state';

import Checkbox from "@material-ui/core/Checkbox";
import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';

const CLASSES = [
  "Algebra",
  "Geometry",
  "Calculus",
  "Computer Programming",
];

const End = () => {
  const id = useStore(state => state.id);
  const [age, setAge] = useState("18");
  const [education, setEducation] = useState("shs");
  const [classesTaken, setClassesTaken] = useState({});
  const [experience, setExperience] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const submit = async () => {
    await apiRequest('save-survey', { id, survey: { age, education, classesTaken, experience } });
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
          <option value="shs">Some high-school</option>
          <option value="hsd">High school diploma</option>
          <option value="sc">Some college</option>
          <option value="cd">College degree</option>
          <option value="gp">Graduate or Professional Degree</option>
        </select>
      </p>
      <p>
        Have you ever taken a course on the subjects below? Select all that apply.
        <FormGroup>
          { CLASSES.map((c, i) =>
            <FormControlLabel
              key={i}
              control={
                <Checkbox
                  checked={classesTaken[c]}
                  onChange={() => setClassesTaken({ ...classesTaken, [c]: !classesTaken[c]})}
                  name={c}
                />
              }
              label={c}
            />
          )}
        </FormGroup>
      </p>
      <p>
        How did you like the experience? Any feedback on how to improve it?
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
              After copying the code, you may close this tab. Have a good rest of your day!
            </p>
          </div>
        : <button onClick={submit}>Finish</button>
      }
    </div>
  );
};
export default End;
