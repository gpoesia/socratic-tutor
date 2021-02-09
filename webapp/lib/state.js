import create from 'zustand';

const useStore = create(
  set => ({
    id: null,
    testAnswers: [],
    exerciseAnswers: [],

    setID: (id) => set(state => ({ id })),
  })
);
export default useStore;
