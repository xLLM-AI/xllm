(function () {
  "use strict";

  if (window.location.hostname !== "xllm.readthedocs.io") {
    return;
  }

  var path = window.location.pathname || "/";
  var targetPath = "/";
  var versionedPath = path.match(/^\/(zh-cn|en)\/(?:latest|stable)\/?(.*)$/);

  if (versionedPath) {
    var language = versionedPath[1] === "zh-cn" ? "zh" : "en";
    var rest = versionedPath[2] || "";
    targetPath = "/" + language + "/" + rest;
  } else if (path !== "/") {
    targetPath = path;
  }

  var target = new URL("https://docs.xllm-ai.com/");
  target.pathname = targetPath;
  target.search = window.location.search;
  target.hash = window.location.hash;

  window.location.replace(target.toString());
})();
