#include <stdio.h>
#include <stdint.h>

int main()
{
  const uint8_t lut_numBnInBnGroups_BG1_R13[30] = {42, 0, 0, 1, 1, 2, 4, 3, 1, 4, 3, 4, 1, 0, 0,
                                                   0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1};

  int total_blocks = 0;
  for (int g = 1; g <= 30; g++) {
    int num_bn = lut_numBnInBnGroups_BG1_R13[g - 1];
    total_blocks += num_bn; //* g;
  }
  printf("// Total inner blocks = %d\n", total_blocks);

  printf("const uint8_t lut_InnerBlock[%d] = {\n", total_blocks);
  int cnt = 0;
  for (int g = 1; g <= 30; g++) {
    int num_bn = lut_numBnInBnGroups_BG1_R13[g - 1];
    for (int bn = 0; bn < num_bn; bn++) {
  
        printf("%d,", g);
        cnt++;
        if (cnt % 20 == 0)
          printf("\n");
      
    }
  }
  printf("\n};\n\n");

  printf("const uint8_t lut_InnerInnerBlock[%d] = {\n", total_blocks);
  cnt = 0;
  for (int g = 1; g <= 30; g++) {
    int num_bn = lut_numBnInBnGroups_BG1_R13[g - 1];
    for (int bn = 0; bn < num_bn; bn++) {
      for (int k = 1; k <= g; k++) {
        printf("%d,", k);
        cnt++;
        if (cnt % 20 == 0)
          printf("\n");
      }
    }
  }
  printf("\n};\n");
  printf("const uint8_t lut_BnIdx[%d] = {\n", total_blocks);
  cnt = 0;
  for (int g = 1; g <= 30; g++) {
    int num_bn = lut_numBnInBnGroups_BG1_R13[g - 1];
    for (int bn = 0; bn < num_bn; bn++) {
      
        printf("%d,", bn + 1); // 从1开始计数
        cnt++;
        if (cnt % 20 == 0)
          printf("\n");
      
    }
  }
  printf("\n};\n");
}