import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import CircularProgress from '@material-ui/core/CircularProgress';
import { apiRequest } from '../lib/api';
import useStore from '../lib/state';

export default function App() {
  const router = useRouter();

  const sessionId = useStore(state => state.id);
  const setID = useStore(state => state.setID);

  useEffect(async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const curriculum = urlParams.get('curriculum');

    if (!sessionId) {
      const { id } = await apiRequest('new-session', { curriculum });
      setID(id);
    }

    router.push('/instructions');
  });

  return (
    <div>
      <p>Starting your session...</p>
      <CircularProgress />
    </div>
  );
}
