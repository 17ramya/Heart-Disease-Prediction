/* Heart Disease Prediction - front-end logic */
(function () {
  "use strict";

  var form = document.getElementById("predict-form");
  var submitBtn = document.getElementById("submit-btn");
  var resultCard = document.getElementById("result");
  var verdictEl = document.getElementById("verdict");
  var meterFill = document.getElementById("meter-fill");
  var probText = document.getElementById("prob-text");
  var barsEl = document.getElementById("categorical-bars");

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    var payload = {};
    var elements = form.elements;
    for (var i = 0; i < elements.length; i++) {
      var el = elements[i];
      if (el.name) {
        payload[el.name] = el.value;
      }
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "Predicting...";

    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (response) {
        return response.json().then(function (data) {
          if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
          }
          return data;
        });
      })
      .then(renderResult)
      .catch(function (error) {
        verdictEl.className = "verdict danger";
        verdictEl.textContent = "Error: " + error.message;
        resultCard.classList.remove("hidden");
      })
      .finally(function () {
        submitBtn.disabled = false;
        submitBtn.textContent = "Predict";
      });
  });

  function renderResult(data) {
    var binary = data.binary;
    var disease = binary.label === 1;

    verdictEl.className = "verdict " + (disease ? "danger" : "safe");
    verdictEl.textContent =
      (disease ? "\u26A0\uFE0F  Heart Disease likely" : "\u2705  No Heart Disease detected") +
      "  \u2014  " +
      binary.prediction;

    var pct = Math.round(binary.probability * 100);
    meterFill.style.width = Math.max(4, pct) + "%";
    probText.textContent =
      "Probability of heart disease: " + pct + "%  (model confidence: " +
      Math.round(binary.confidence * 100) + "%)";

    barsEl.innerHTML = "";
    var probs = data.categorical.probabilities;
    Object.keys(probs).forEach(function (label) {
      var value = probs[label];
      var li = document.createElement("li");

      var name = document.createElement("span");
      name.className = "name";
      name.textContent = label;

      var track = document.createElement("div");
      track.className = "track";
      var fill = document.createElement("div");
      fill.className = "fill";
      fill.style.width = Math.round(value * 100) + "%";
      track.appendChild(fill);

      var val = document.createElement("span");
      val.className = "val";
      val.textContent = Math.round(value * 100) + "%";

      li.appendChild(name);
      li.appendChild(track);
      li.appendChild(val);
      barsEl.appendChild(li);
    });

    resultCard.classList.remove("hidden");
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
})();
