import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { applyTokens } from "./design/tokens";
import { coreLink } from "./services/core";
import "./styles.css";

applyTokens(document.documentElement);
coreLink.start();

const root = document.getElementById("root");
if (root === null) throw new Error("Falta #root en index.html");
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
