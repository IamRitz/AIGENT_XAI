\ Model Model
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  epsilon_max + adder + epsilon_max_2
Subject To
 R0: C1 = 1
 R1: - epsilon_max + C2 <= 0
 R2: epsilon_max + C2 >= 0
 R3: - epsilon_max + C3 <= 0
 R4: epsilon_max + C3 >= 0
 R5: - epsilon_max + C4 <= 0
 R6: epsilon_max + C4 >= 0
 R7: - epsilon_max + C5 <= 0
 R8: epsilon_max + C5 >= 0
 R9: - epsilon_max + C6 <= 0
 R10: epsilon_max + C6 >= 0
 R11: - epsilon_max + C7 <= 0
 R12: epsilon_max + C7 >= 0
 R13: - epsilon_max + C8 <= 0
 R14: epsilon_max + C8 >= 0
 R15: - epsilon_max + C9 <= 0
 R16: epsilon_max + C9 >= 0
 R17: C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + epsilon_max_2 >= 0
 R18: C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 - epsilon_max_2 <= 0
 qc0: - 2 C1 - adder + [ - C1 * C2 + C1 * C3 ] >= 0
Bounds
 epsilon_max <= 5
 0.5 <= adder <= 100000
 epsilon_max_2 <= 5
End
