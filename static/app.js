function startClock() {
  const clock = document.querySelector("[data-live-clock]");
  if (!clock) return;

  const render = () => {
    clock.textContent = new Date().toLocaleString();
  };

  render();
  window.setInterval(render, 1000);
}

function bindSnapshotModes() {
  ["home", "away"].forEach((side) => {
    const modeSelect = document.querySelector(`[data-snapshot-mode="${side}"]`);
    const monthField = document.querySelector(`[data-month-field="${side}"]`);
    const monthInput = monthField?.querySelector("input[type='month']");
    if (!modeSelect || !monthField || !monthInput) return;

    const update = () => {
      const historical = modeSelect.value === "historical";
      monthField.dataset.disabled = historical ? "false" : "true";
      monthInput.disabled = !historical;
      monthInput.required = historical;
      if (!historical) monthInput.value = "";
    };

    modeSelect.addEventListener("change", update);
    update();
  });
}

function bindPredictForm() {
  const form = document.querySelector("[data-predict-form]");
  const submitButton = document.querySelector("[data-submit-button]");
  if (!form || !submitButton) return;

  form.addEventListener("submit", (event) => {
    const homeTeam = form.querySelector('select[name="home_team"]')?.value;
    const awayTeam = form.querySelector('select[name="away_team"]')?.value;
    if (homeTeam && awayTeam && homeTeam === awayTeam) {
      event.preventDefault();
      window.alert("Choose two different clubs for the comparison.");
      return;
    }

    submitButton.disabled = true;
    submitButton.textContent = "Predicting...";
  });
}

async function refreshTimeline() {
  const timeline = document.querySelector("[data-timeline]");
  if (!timeline) return;

  try {
    const response = await fetch("/api/timeline");
    if (!response.ok) return;
    const payload = await response.json();
    if (!payload.items?.length) return;

    timeline.innerHTML = payload.items
      .map(
        (item, index) => `
          <a class="timeline-item" href="/predictions/${item.id}">
            <div class="timeline-minute">${index + 1}</div>
            <div class="timeline-content">
              <div class="timeline-title">${item.home_team} vs ${item.away_team}</div>
              <div class="timeline-snapshots">${item.home_snapshot_label} | ${item.away_snapshot_label}</div>
              <div class="timeline-meta">
                <span>Prediction ${item.prediction}</span>
                <span>${item.created_at}</span>
              </div>
            </div>
          </a>
        `
      )
      .join("");
  } catch (_) {
    // Keep the current timeline when refresh fails.
  }
}

startClock();
bindSnapshotModes();
bindPredictForm();
refreshTimeline();
window.setInterval(refreshTimeline, 15000);
