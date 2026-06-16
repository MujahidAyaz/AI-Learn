import numpy as np

# ---------------------------
# Dataset
# ---------------------------
marks = np.array([
    [85, 90, 78],
    [70, 88, 92],
    [95, 76, 89],
    [60, 65, 70],
    [88, 91, 84]
])

subjects = ["Math", "Physics", "Chemistry"]

# ---------------------------
# Task 1: Array Information
# ---------------------------
print("=== ARRAY INFO ===")
print("Shape:", marks.shape)
print("Dimensions:", marks.ndim)
print("Size:", marks.size)
print("Data Type:", marks.dtype)

print("\n")

# ---------------------------
# Task 2: Average per Student
# axis=1 → row-wise
# ---------------------------
print("=== STUDENT AVERAGES ===")

student_avg = np.mean(marks, axis=1)

for i, avg in enumerate(student_avg, start=1):
    print(f"Student {i}: {avg:.2f}")

print("\n")

# ---------------------------
# Task 3: Average per Subject
# axis=0 → column-wise
# ---------------------------
print("=== SUBJECT AVERAGES ===")

subject_avg = np.mean(marks, axis=0)

for i, avg in enumerate(subject_avg):
    print(f"{subjects[i]} Average: {avg:.2f}")

print("\n")

# ---------------------------
# Task 4: Highest in Each Subject
# ---------------------------
print("=== HIGHEST MARKS ===")

subject_max = np.max(marks, axis=0)

for i, val in enumerate(subject_max):
    print(f"{subjects[i]} Highest: {val}")

print("\n")

# ---------------------------
# Task 5: Lowest in Each Subject
# ---------------------------
print("=== LOWEST MARKS ===")

subject_min = np.min(marks, axis=0)

for i, val in enumerate(subject_min):
    print(f"{subjects[i]} Lowest: {val}")

print("\n")

# ---------------------------
# Task 6: Top Student
# ---------------------------
print("=== TOP STUDENT ===")

total_marks = np.sum(marks, axis=1)
top_student_index = np.argmax(total_marks)

print(f"Top Student: Student {top_student_index + 1}")
print(f"Total Marks: {total_marks[top_student_index]}")

print("\n")

# ---------------------------
# Task 7: Add Grace Marks
# ---------------------------
print("=== GRACE MARKS ADDED ===")

new_marks = marks + 5
print(new_marks)

print("\n")

# ---------------------------
# Task 8: Passed Students
# ---------------------------
print("=== PASSED STUDENTS ===")

student_avg = np.mean(marks, axis=1)
passed = student_avg >= 75

for i in range(len(passed)):
    if passed[i]:
        print(f"Student {i + 1}")

print("\n")

# ---------------------------
# Bonus: Grade Function
# ---------------------------
def get_grade(avg):
    if avg >= 90:
        return "A"
    elif avg >= 80:
        return "B"
    elif avg >= 70:
        return "C"
    elif avg >= 60:
        return "D"
    else:
        return "F"

print("=== GRADES ===")

for i, avg in enumerate(student_avg, start=1):
    grade = get_grade(avg)
    print(f"Student {i}: {avg:.2f} → Grade {grade}")