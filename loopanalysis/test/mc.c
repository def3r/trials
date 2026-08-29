int mc(int* edges[2], int ne, int* partition) {
  int cut = 0;
  for (int i = 0; i < ne; i++) {
    int u = edges[i][0];
    int v = edges[i][1];
    if (partition[u] != partition[v]) {
      cut++;
    }
  }
  return cut;
}
