import os
import re


def srt_to_plain_text(file_path):
    """
    Converts an SRT subtitles file into plain text by removing:
    - Subtitle numbers
    - Timings
    - HTML tags
    """
    # Try to read the file with UTF-8 and ISO-8859-1 encodings
    for encoding in ['utf-8', 'iso-8859-1']:
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                srt_content = file.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"Error: Unable to decode file {file_path}")
        return ""

    # Remove subtitle numbers and timings
    srt_content = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', srt_content)

    # Remove any remaining HTML tags
    srt_content = re.sub(r'<[^>]+>', '', srt_content)

    # Remove empty lines and merge all lines into a single string
    plain_text = ' '.join(line.strip() for line in srt_content.splitlines() if line.strip())

    return plain_text


def process_all_srt_files(input_folder, output_folder):
    """
    Processes all SRT files in the input folder and saves their cleaned content
    to the output folder with a .txt extension.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    for file_name in os.listdir(input_folder):
        # Process only .srt files
        if file_name.lower().endswith('.srt'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_name = os.path.splitext(file_name)[0] + '.txt'
            output_file_path = os.path.join(output_folder, output_file_name)

            # Convert SRT to plain text
            cleaned_text = srt_to_plain_text(input_file_path)

            if cleaned_text:  # Only write if there's content
                # Save the output to the new folder
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(cleaned_text)

                print(f"Processed: {file_name} --> {output_file_name}")
            else:
                print(f"Skipped: {file_name} (Unable to process)")


# Input folder and output folder paths
input_folder = "../data/ml-20m-psm/subtitles"  # Replace with your input folder path
output_folder = "../data/processed/subtitles"  # Replace with your output folder path

# Process all SRT files in the input folder
process_all_srt_files(input_folder, output_folder)