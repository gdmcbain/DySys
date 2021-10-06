#!/usr/bin/env stack
{-stack runhaskell
 --package shake
 -}

import Development.Shake
import Development.Shake.FilePath
import Development.Shake.Clean (cleanRules)

main :: IO ()
main = shakeArgs shakeOptions $ do
  want ["driver.png"]

  cleanRules ["*~", "*.png"]

  "*.png" %> \png -> do
    let script = png -<.> "py"
    need [script]
    pyFlags <- getEnv "PYFLAGS"
    cmd "python" pyFlags [script]
