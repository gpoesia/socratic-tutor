import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import useStore from "../lib/state";
import { useRouter } from "next/router";
import { apiRequest } from "../lib/api";
import { Button } from "@material-ui/core";
import _ from "lodash";
import CircularProgress from "@material-ui/core/CircularProgress";

const TestQuestionsJson = require("./turing_test.json");

const RankSolutions = dynamic(
  () => import("../lib/components/rank-solutions.js"),
  { ssr: false }
);

const TuringTest = () => {
  const router = useRouter();
  const sessionId = useStore((state) => state.id);

  const [TestQuestions, setTestQuestions] = useState(null);
  const [checked, setChecked] = useState(new Set());
  const [currQIdx, setQIdx] = useState(0);

  useEffect(() => {
    //load questions from json file. shuffle questions.
    if (TestQuestions === null) {
      let questions = TestQuestionsJson["experiment"]
        .map((a) => ({ sort: Math.random(), value: a }))
        .sort((a, b) => a.sort - b.sort)
        .map((a) => a.value);
      setTestQuestions(questions);
    }
  });

  const canAdvance = checked.size == 2;
  const next = async () => {
    //first save answers for current problem
    await apiRequest("save-answers", {
      id: sessionId,
      qStr: TestQuestions[currQIdx]["question"],
      qAns: Array.from(checked),
    });
    //then go to the next problem/next page
    if (currQIdx + 1 == TestQuestions.length) {
      router.push("/end");
    } else {
      setQIdx(currQIdx + 1);
      setChecked(new Set());
    }
  };

  if (TestQuestions === null) {
    return (
      <div className="content">
        <CircularProgress />
      </div>
    );
  }

  return (
    <div className="content">
      <h1>Human or Machine: Step-by-Step Solutions for Equation Problems</h1>
      <p>Consider this equation problem,</p>
      <p>{TestQuestions[currQIdx]["question"]}</p>
      <p>
        The following are 4 step-by-step solutions, of which two are written by
        human. Please pick the two solutions that you believe are most likely to
        be written by human.
      </p>
      <RankSolutions
        solutions={TestQuestions[currQIdx]["solutions"]}
        checked={checked}
        setChecked={setChecked}
      />
      <p>
        To pick a solution, please check the box. You must pick exactly two
        solutions.
      </p>
      <div className="center-button">
        <Button
          onClick={() => {
            if (canAdvance) next();
            else alert("You must select two solutions.");
          }}
        >
          Continue
        </Button>
      </div>
    </div>
  );
};

export default TuringTest;
