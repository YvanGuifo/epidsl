"use strict";

const { LanguageClient, TransportKind } = require("vscode-languageclient/node");
const vscode = require("vscode");

let client;

function activate(context) {
  const cfg = vscode.workspace.getConfiguration("epidsl");
  const cmd = cfg.get("serverCommand", ["epidsl", "server"]);

  const serverOptions = {
    command: cmd[0],
    args: cmd.slice(1),
    transport: TransportKind.stdio,
  };

  const clientOptions = {
    documentSelector: [{ scheme: "file", language: "epidsl" }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher("**/*.epidsl"),
    },
  };

  client = new LanguageClient("epidsl", "EpiDSL Language Server", serverOptions, clientOptions);
  client.start();
}

function deactivate() {
  if (client) return client.stop();
}

module.exports = { activate, deactivate };
