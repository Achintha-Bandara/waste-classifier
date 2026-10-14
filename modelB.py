from collections import Counter

class_counts = Counter(labels)

print("\nImages per class:")
for class_name, class_idx in class_to_idx.items():
    print(f"{class_name:30s}: {class_counts[class_idx]}")

print("\nLargest classes:")
for class_idx, count in class_counts.most_common(10):
    print(f"{class_names[class_idx]:30s}: {count}")