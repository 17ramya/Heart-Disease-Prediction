/* Heart Disease Prediction - front-end logic */
(function () {
  "use strict";

  var LABELS = window.FEATURE_LABELS || {};

  function el(id) { return document.getElementById(id); }
  function pct(x) { return Math.round((x || 0) * 100); }
  function color(disease) { return disease ? "#dc2626" : "#16a34a"; }
  function label(key) { return LABELS[key] || key; }

  /* ---------------- Theme (light mode by default) ---------------- */
  var THEME_KEY = "heartrisk-theme";

  function currentTheme() {
    try { return localStorage.getItem(THEME_KEY) || "light"; }
    catch (e) { return "light"; }
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    var btn = el("theme-toggle");
    if (btn) {
      btn.textContent = theme === "dark" ? "\u263E" : "\u2600";
      btn.setAttribute(
        "title",
        theme === "dark" ? "Switch to light mode" : "Switch to dark mode"
      );
    }
  }

  applyTheme(currentTheme());

  var themeToggle = el("theme-toggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", function () {
      var isDark =
        document.documentElement.getAttribute("data-theme") === "dark";
      var next = isDark ? "light" : "dark";
      try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* ignore */ }
      applyTheme(next);
    });
  }

  /* ---------------- Start with a clean slate ---------------- */
  /* The application boxes must never show a previous patient's data, not even
     after a refresh or a back/forward navigation (browser form restoration). */
  function clearForm() {
    var node = el("predict-form");
    if (!node) { return; }
    var elements = node.elements;
    for (var i = 0; i < elements.length; i++) {
      var field = elements[i];
      if (!field.name) { continue; }
      if (field.tagName === "SELECT") {
        field.value = "";
        field.selectedIndex = 0; // back to the "Select..." placeholder
      } else {
        field.value = "";
      }
    }
  }

  function clearResults() {
    var results = el("results");
    if (results) { results.classList.add("hidden"); }
    ["model-cards", "categorical-bars", "risk-factors"].forEach(function (id) {
      var node = el(id);
      if (node) { node.innerHTML = ""; }
    });
  }

  clearForm();
  clearResults();
  window.addEventListener("pageshow", function (event) {
    if (event.persisted) {
      clearForm();
      clearResults();
    }
  });

  /* ---------------- Navigation menu ---------------- */
  var navToggle = el("nav-toggle");
  var navLinks = el("nav-links");
  navToggle.addEventListener("click", function () {
    navLinks.classList.toggle("open");
  });
  navLinks.addEventListener("click", function (e) {
    if (e.target.tagName === "A") { navLinks.classList.remove("open"); }
  });

  var targetIds = ["predict", "performance", "dataset", "api", "about"];
  var navAnchors = Array.prototype.slice.call(document.querySelectorAll(".nav-links a"));
  var toTop = el("to-top");

  function onScroll() {
    var pos = window.scrollY + 100;
    var current = "";
    targetIds.forEach(function (id) {
      var node = el(id);
      if (node && node.offsetTop <= pos) { current = id; }
    });
    navAnchors.forEach(function (a) {
      a.classList.toggle("active", a.getAttribute("href") === "#" + current);
    });
    toTop.classList.toggle("show", window.scrollY > 520);
  }
  window.addEventListener("scroll", onScroll);
  onScroll();
  toTop.addEventListener("click", function () {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  /* ---------------- Prediction form ---------------- */
  var form = el("predict-form");
  var submitBtn = el("submit-btn");
  var results = el("results");
  var verdictEl = el("verdict");
  var gaugeValue = el("gauge-value");
  var gauge = gaugeValue.parentElement;
  var meterFill = el("meter-fill");
  var probText = el("prob-text");
  var modelCards = el("model-cards");
  var barList = el("categorical-bars");
  var factorList = el("risk-factors");

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    var payload = {};
    var elements = form.elements;
    for (var i = 0; i < elements.length; i++) {
      if (elements[i].name) { payload[elements[i].name] = elements[i].value; }
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "Predicting...";

    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (response) {
        if (response.status === 401) {
          window.location.href = "/login";
          throw new Error("Session expired - please sign in again.");
        }
        return response.json().then(function (data) {
          if (!response.ok) { throw new Error(data.error || "Prediction failed."); }
          return data;
        });
      })
      .then(renderResult)
      .catch(function (error) {
        verdictEl.className = "verdict danger";
        verdictEl.textContent = "Error: " + error.message;
        results.classList.remove("hidden");
      })
      .finally(function () {
        submitBtn.disabled = false;
        submitBtn.textContent = "Predict risk";
      });
  });

  function renderResult(data) {
    var binary = data.binary;
    var disease = binary.label === 1;
    var risk = pct(binary.probability);

    verdictEl.className = "verdict " + (disease ? "danger" : "safe");
    verdictEl.textContent =
      (disease ? "\u26A0\uFE0F  Heart Disease likely" : "\u2705  No Heart Disease detected") +
      "  \u2014  " + binary.prediction;

    gaugeValue.textContent = risk;
    gaugeValue.style.color = color(disease);
    gauge.style.setProperty("--p", risk);
    gauge.style.setProperty("--gcolor", color(disease));

    meterFill.style.width = Math.max(3, risk) + "%";
    probText.textContent =
      "Ensemble risk probability: " + risk + "%  \u00b7  confidence " +
      pct(binary.confidence) + "%";

    /* per-model cards */
    modelCards.innerHTML = "";
    data.models.forEach(function (model) {
      var isDisease = model.probability >= 0.5;
      var card = document.createElement("div");
      card.className = "model-card";

      var head = document.createElement("div");
      head.className = "mc-head";
      var name = document.createElement("span");
      name.className = "mc-name";
      name.textContent = model.label;
      var badge = document.createElement("span");
      badge.className = "mc-badge " + (isDisease ? "disease" : "healthy");
      badge.textContent = isDisease ? "Disease" : "Healthy";
      head.appendChild(name);
      head.appendChild(badge);

      var bar = document.createElement("div");
      bar.className = "mc-bar";
      var fill = document.createElement("span");
      fill.style.width = Math.max(3, pct(model.probability)) + "%";
      fill.style.background = color(isDisease);
      bar.appendChild(fill);

      var meta = document.createElement("div");
      meta.className = "mc-meta";
      var riskTxt = document.createElement("span");
      riskTxt.textContent = "risk " + pct(model.probability) + "%";
      var accTxt = document.createElement("span");
      accTxt.textContent = model.accuracy != null ? "acc " + pct(model.accuracy) + "%" : "";
      meta.appendChild(riskTxt);
      meta.appendChild(accTxt);

      card.appendChild(head);
      card.appendChild(bar);
      card.appendChild(meta);
      modelCards.appendChild(card);
    });

    /* severity distribution */
    barList.innerHTML = "";
    var probs = data.categorical.probabilities;
    Object.keys(probs).forEach(function (name) {
      barList.appendChild(makeBar(name, probs[name], "fill"));
    });

    /* key risk factors */
    factorList.innerHTML = "";
    var factors = data.risk_factors || [];
    var maxAbs = factors.reduce(function (m, f) {
      return Math.max(m, Math.abs(f.contribution));
    }, 0.0001);
    factors.forEach(function (f) {
      var width = Math.max(4, Math.round((Math.abs(f.contribution) / maxAbs) * 100));
      var cls = f.increases_risk ? "fill up" : "fill down";
      var sign = f.increases_risk ? "+" : "\u2212";
      factorList.appendChild(
        makeBar(label(f.feature), width, cls, sign + Math.abs(f.contribution).toFixed(2))
      );
    });

    results.classList.remove("hidden");
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* list item builder: name | track + fill | value */
  function makeBar(name, value, fillClass, valueText) {
    var li = document.createElement("li");

    var nameEl = document.createElement("span");
    nameEl.className = "name";
    nameEl.textContent = name;

    var track = document.createElement("div");
    track.className = "track";
    var fill = document.createElement("div");
    fill.className = fillClass;
    var width = value <= 1 ? Math.round(value * 100) : value;
    fill.style.width = Math.max(4, width) + "%";
    track.appendChild(fill);

    var val = document.createElement("span");
    val.className = "val";
    val.textContent = valueText != null ? valueText : Math.round(value * 100) + "%";

    li.appendChild(nameEl);
    li.appendChild(track);
    li.appendChild(val);
    return li;
  }
})();

