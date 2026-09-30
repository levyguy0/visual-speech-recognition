import os

print(
    [
        len(
            [
                d
                for d in os.listdir(f"data/all_grid_word_clips/{dir}")
                if d != ".DS_Store"
            ]
        )
        for dir in os.listdir("data/all_grid_word_clips")
        if dir != ".DS_Store"
    ]
)
