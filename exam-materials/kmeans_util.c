#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdarg.h>
#include <ctype.h>
#include <stdint.h>
#include <math.h>

/*
 * kmeans_util.c
 * -------------
 * Tiny utility file shared by serial and MPI K-means.
 *
 * The key helper here is filestats(), which scans a text data file to determine:
 * - total number of whitespace-separated tokens
 * - total number of lines
 *
 * Why that helps:
 * - K-means input is stored as text.
 * - Counting tokens/lines lets the program infer dataset dimensions before doing
 *   the full parse.
 */

// Sets number of lines and total number of whitespace separated
// tokens in the file. Returns -1 if file can't be opened, 0 on
// success.
//
// EXAMPLE: int ret = filestats("digits_all_1e1.txt", &toks, &lines);
// toks  is now 7860 : 10 lines with 786 tokens per line, label + ":" + 28x28 pixels
// lines is now 10   : there are 10 lines in the file
int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines){
  FILE *fin = fopen(filename,"r");
  if(fin == NULL){
    printf("Failed to open file '%s'\n",filename);
    return -1;
  }

  ssize_t ntokens=0, nlines=0, column=0;
  int intoken=0, token;

  /*
   * Scan one character at a time and count transitions into tokens.
   * A token starts when we move from whitespace to non-whitespace.
   */
  while((token = fgetc(fin)) != EOF){
    if(token == '\n'){
      column = 0;
      nlines++;
    }
    else{
      column++;
    }

    if(isspace(token) && intoken==1){
      intoken = 0;
    }
    else if(!isspace(token) && intoken==0){
      intoken = 1;
      ntokens++;
    }
  }

  /* Account for the last line if the file does not end with newline. */
  if(column != 0){
    nlines++;
  }

  *tot_tokens = ntokens;
  *tot_lines = nlines;
  fclose(fin);
  return 0;
}
