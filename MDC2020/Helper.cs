using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MDC2020
{
    public static class Helper
    {
        /// <summary>
        /// Moves up number of specified directories up to root directory
        /// </summary>
        /// <param name="path">starting point</param>
        /// <param name="moveUpLevels">number of levels to move up in directory structure</param>
        /// <returns>New directory location or root if number of levels exceeds number of possible moves</returns>
        public static string NavigateUpDirectory(string path, int moveUpLevels)
        {
            var newDirectory = path;
            for (int level = 0; level < moveUpLevels; level++)
            {
                if (Directory.GetDirectoryRoot(path) == newDirectory)
                    continue;
                newDirectory = Directory.GetParent(newDirectory).FullName;
            }

            return newDirectory;
        }

    }
}
