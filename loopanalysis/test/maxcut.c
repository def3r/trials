// $ clang -emit-llvm -S -Xclang -disable-O0-optnone maxcut.c -o maxcut_raw.ll
// $ opt -load-pass-plugin=../build/MinPass.so -passes="min-pass" -disable-output maxcut.ll

int actual() {
  int cut = 0;

  return cut;
}

int maxcut_base(int edges[][2], int num_edges, int* partition) {
  int cut = 0;
  for (int i = 0; i < num_edges; i++) {
    int u = edges[i][0];
    int v = edges[i][1];
    if (partition[u] != partition[v])
      cut++;
  }
  return cut;
}

int maxcut1(int edges[][2], int num_edges, int* partition) {
  int cut = 0;
  for (int i = 0; i < num_edges; i++) {
    if (partition[edges[i][0]] != partition[edges[i][1]])
      cut++;
  }
  return cut;
}

int maxcut2(int edges[][2], int cut, int num_edges, int* partition) {
  for (int i = 0; i < num_edges; i++) {
    if (partition[edges[i][0]] != partition[edges[i][1]])
      cut++;
  }
  return cut;
}

int maxcut3(int edges[][2], int cut, int num_edges, int* partition) {
  for (int i = num_edges; i >= 0; i--) {
    if (partition[edges[i][0]] != partition[edges[i][1]])
      cut++;
  }
  return cut;
}

int count_mismatched_grades(int pairs[][2], int num_pairs, int* grade_level) {
  int mismatches;
  for (int i = 0; i < num_pairs; i++) {
    mismatches =
        mismatches + (grade_level[pairs[i][0]] != grade_level[pairs[i][1]]);
  }
  return mismatches;
}
