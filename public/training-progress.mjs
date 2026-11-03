const phases = [[0, 'warm'], [1, 'ep2'], [3, 'ep3']];

// Legacy backups only saved the active phase. Completed earlier phases used
// their saved targets; retain the active epoch, shuffled order and cursor.
export function rememberPosition(state) {
  if (!state.phaseProgress) {
    state.phaseProgress = Object.fromEntries(phases.map(([phase, key]) => {
      const completed = Math.max(0, ...(state.history || []).filter(h => h.phase === phase).map(h => h.epoch));
      return [phase, {epoch: phase < state.phase ? Math.max(completed, state.config[key]) : 0, cursor: 0, order: []}];
    }));
    state.calibrationUpdates = state.phase >= 2 ? state.updates2 : null;
  }
  if (state.phaseProgress[state.phase]) {
    state.phaseProgress[state.phase] = {epoch: state.epoch, cursor: state.cursor, order: state.order};
  }
}

export function setEpochTargets(state, targets) {
  for (const [, key] of phases) {
    if (!Number.isInteger(targets[key]) || targets[key] < 1) throw Error('エポック数は1以上の整数を指定してください');
  }
  rememberPosition(state);
  for (const [phase, key] of phases) {
    const position = state.phaseProgress[phase];
    state.config[key] = Math.max(targets[key], position.epoch + (position.cursor > 0 ? 1 : 0));
  }
}

// Return true only when new 2D updates require fresh calibration residuals.
export function selectNextPhase(state) {
  rememberPosition(state);
  let phase = phases.find(([p, key]) => state.phaseProgress[p].epoch < state.config[key])?.[0] ?? 4;
  let resetCalibration = false;
  if (phase >= 3 && (state.calibrationUpdates !== state.updates2 || state.calibrationIndex < 512)) {
    phase = 2;
    if (state.calibrationUpdates !== state.updates2) {
      state.calibrationUpdates = state.updates2;
      state.calibrationIndex = 0;
      resetCalibration = true;
    }
  }
  state.phase = phase;
  const position = state.phaseProgress[phase];
  state.epoch = position?.epoch ?? 0;
  state.cursor = position?.cursor ?? 0;
  state.order = position?.order ?? [];
  return resetCalibration;
}
