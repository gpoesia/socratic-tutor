import React, { useState, useEffect, useRef } from 'react';
import IconButton from '@material-ui/core/IconButton';
import Backspace from '@material-ui/icons/Backspace';
import styles from '../../styles/tutor-dialogue.module.css';

const useLog = (initial) => {
  initial = initial || [];

  const [list, set] = useState(initial);

  return [list, (item, l) => {
    const newList = (l || list).concat([item]);
    set(newList);
    return newList;
  }];
}

export default function TutorDialogue(props) {
  const [facts, setFacts] = useState(props.problem.facts);
  const [messages, addMessage] = useLog(makeInitialMessages(props.problem));
  const [input, setInput] = useState("");
  const messagesRef = useRef(null);

  useEffect(() => {
    messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
  });

  const eraseFact = (i) => {
    const remainingFacts = facts.slice(0, i).concat(facts.slice(i+1));
    const erasedFact = facts[i];
    console.log(erasedFact);
    addMessage({ author: 'student', message: `Erase (${i+1}) ${erasedFact}` });
    setFacts(remainingFacts);
  };

  return (
    <div className={styles.dialogueContainer}>
      <div className={styles.historyContainer}>
        <h1>Dialogue</h1>

        <div className={styles.messagesContainer} ref={messagesRef}>
          { messages.map((m, i) =>
            <div className={styles['message-'+ m.author]} key={i}>
              <pre>{m.message}</pre>
            </div>
          )}
        </div>

        <div className={styles.inputContainer}>
          <input type="text" value={input}
                 onChange={e => setInput(e.target.value)} />
          <button onClick={() => parseTerm(input, addMessage,
                                           facts, setFacts,
                                           setInput)}>
            Send
          </button>
          <button onClick={() => askLeadingQuestion(facts, props.problem.goals, addMessage)}>Help</button>
          <button onClick={() => check(facts, props.problem.goals, addMessage)}>Check</button>
        </div>
      </div>

      <div className={styles.factsContainer}>
        <h1>Progress</h1>
        <ol className={styles.factsList}>
          { facts.map((f, i) =>
              <li key={i}>
                {f}
                { i >= props.problem.facts.length &&
                  <IconButton
                    color="primary"
                    aria-label="upload picture"
                    component="span"
                    onClick={() => eraseFact(i)}
                  >
                    <Backspace />
                  </IconButton>
                }
              </li>
            )
          }
        </ol>
      </div>
    </div>
  );
}

function makeInitialMessages(problem) {
  const m =
    [
      { author: 'tutor',
        message:
          'Let\'s solve a problem! Given:\n' +
          problem.facts.map((f, i) => ` (${i + 1}) ${f}`).join('\n') +
          '\nYou need to solve ' +
          problem.goals.map((g, i) => g).join(', ')
      }
    ];

  return m;
}

async function tutorRequest(endpoint, parameters) {
  const req = await fetch('/api/' + endpoint + '?params=' + encodeURIComponent(JSON.stringify(parameters)));
  console.log('Fetched API:', req);
  const res = await req.json();
  return res;
}

async function parseTerm(term, addMessage, facts, setFacts, setInput) {
  let l = addMessage({ author: 'student', message: term });

  try {
    const response = await tutorRequest('parse-term', { 'term': term });
    console.log('Got:', response);

    const parsedTerm = response.value;

    if (facts.indexOf(parsedTerm) == -1) {
      setFacts(facts.concat([parsedTerm]));
    }
  } catch (e) {
    addMessage({ author: 'tutor', message: 'Sorry, I could not understand that. Maybe check the formatting?' }, l);
  }

  setInput('');
}

async function askLeadingQuestion(facts, goals, addMessage) {
  let l = addMessage({ author: 'student', message: 'Help!' });
  l = addMessage({ author: 'tutor', message: 'Hmm, let me think...' }, l);

  const response = await tutorRequest('leading-question', { facts, goals });
  addMessage({ author: 'tutor', message: response }, l);
}


async function check(facts, goals, addMessage) {
  let l = addMessage({ author: 'student', message: 'Check' });
  l = addMessage({ author: 'tutor', message: 'All right, let me take a look...' }, l);

  const response = await tutorRequest('check', { facts, goals });

  if (response == 'correct') {
    addMessage({ author: 'tutor', message: 'Looks correct so far!' });
  } else if (response == 'correct-finished') {
    addMessage({ author: 'tutor', message: 'That\'s it, you\'re done!' });
  } else if (typeof response == 'string') {
    addMessage({ author: 'tutor', message: response });
  }

  console.log(response);
}
