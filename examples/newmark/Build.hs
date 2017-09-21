#!/usr/bin/env stack
{- stack runhaskell
 --package shake
 -}

import Development.Shake
import Development.Shake.FilePath

main :: IO ()
main = shakeArgs shakeOptions $ do
  want ["ph.png"]

  "clean" ~> do
    need ["hlint", "PEP-8"]
    removeFilesAfter "." ["msm*", "*~", "*.png"]

  "hlint" ~> cmd "hlint" "."

  "PEP-8" ~> cmd "flake8" "."

  "*.png" %> \png -> do
    let script = png -<.> "py"
    need [script]
    pyFlags <- getEnv "PYFLAGS"
    cmd "python" pyFlags script
