import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import json

# --- Global variables to store state ---
line_names = ['l', 'l_prime', 'l_double_prime']
point_names = ['p', 'p_prime', 'p_double_prime']
# We need to store the axis and figure to update them
fig = None
ax = None

class RankComputation():
    def __init__(self, data_directory, correspondence_type):
        self.data_directory = data_directory
        self.correspondence_type = correspondence_type
        self.correspondence_types = {
            'lll': [self.line_line_line,
                    self.onclick_line_line_line],
            'ppp': [self.point_point_point,
                    self.onclick_point_point_point]
        }
        self.image_paths = self.get_image_names()
        self.triple = []
        self.run_function, self.onclick_function = self.configure_correspondence_type(correspondence_type)

    def configure_correspondence_type(self, correspondence_type):
        attributes = (self.correspondence_types.get(correspondence_type))
        run_function, onclick_function = attributes
        return run_function, onclick_function

    def run(self):
        self.run_function()

    def save_lines_to_file(self):
        """Saves the calculated line equations to a JSON file."""
        filename = 'line_line_line.json'
        output_path = os.path.join(self.data_directory, filename)
        print(f"\nSaving 3 lines to {filename}...")
        try:
            with open(output_path, 'w') as f:
                json.dump({
                    "l": list(self.triple[0].tolist()),
                    "l_prime": list(self.triple[1].tolist()),
                    "l_double_prime": list(self.triple[2].tolist())
                }, f)
            print("File saved successfully.")
        except IOError as e:
            print(f"Error saving file: {e}")

    def save_points_to_file(self):
        """Saves the calculated points to a JSON file."""
        filename = 'point_point_point.json'
        output_path = os.path.join(self.data_directory, filename)
        print(f"\nSaving 3 points to {filename}...")
        try:
            with open(output_path, 'w') as f:
                json.dump({
                    "p": list(self.triple[0].tolist()),
                    "p_prime": list(self.triple[1].tolist()),
                    "p_double_prime": list(self.triple[2].tolist())
                }, f)
            print("File saved successfully.")
        except IOError as e:
            print(f"Error saving file: {e}")

    def onclick_line_line_line(self, event, plot_state):
        """
        Main event handler for mouse clicks.
        'plot_state' is a dictionary containing state for *this plot only*.
        """
        # Unpack the state for this specific plot
        fig = plot_state["fig"]
        ax = plot_state["ax"]
        line_name = plot_state["line_name"]
        cid = plot_state["cid"]

        # Ignore clicks outside the plot axes
        if not event.inaxes:
            return

        # Get click coordinates
        x, y = event.xdata, event.ydata
        
        # Add click as a homogeneous point
        p_homogeneous = np.array([x, y, 1.0])
        plot_state["points"].append(p_homogeneous)
        
        # print(f"  Point {len(plot_state['points'])} clicked at: ({x:.2f}, {y:.2f})")
        
        # Plot a marker at the click location
        ax.plot(x, y, 'rx', markersize=8)

        if len(plot_state["points"]) == 1:
            # Only one point clicked so far
            ax.set_title(f"Click point 2 for line '{line_name}'")
        
        elif len(plot_state["points"]) == 2:
            # This is the second click, we can define the line
            p1 = plot_state["points"][0]
            p2 = plot_state["points"][1]
            
            # Calculate the line using the cross product
            line_eq = np.cross(p1, p2)
            
            # Normalize the line equation
            norm_factor = np.sqrt(line_eq[0]**2 + line_eq[1]**2)
            if norm_factor > 1e-6:
                line_eq = line_eq / norm_factor
            
            # print(f"  Line '{line_name}' calculated: {line_eq}")
            
            # Add the result to our *global* list
            self.triple.append(line_eq)
                        
            # Update title
            ax.set_title(f"Line '{line_name}' defined. Close this window to continue.")
            
            # We are done with this plot. Disconnect the event handler
            # and close the window.
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig) # This un-blocks the 'plt.show()' call in main()
        
        # Redraw the plot
        fig.canvas.draw()

    def onclick_point_point_point(self, event, plot_state):
        """
        Main event handler for mouse clicks.
        'plot_state' is a dictionary containing state for *this plot only*.
        """
        # Unpack the state for this specific plot
        fig = plot_state["fig"]
        ax = plot_state["ax"]
        point_name = plot_state["point_name"]
        cid = plot_state["cid"]

        # Ignore clicks outside the plot axes
        if not event.inaxes:
            return

        # --- This is the first and only click ---
        
        # Get click coordinates
        x, y = event.xdata, event.ydata
        
        # Create homogeneous point
        p_homogeneous = np.array([x, y, 1.0], dtype=np.int32)
        
        print(f"  Point '{point_name}' clicked at: ({x:.2f}, {y:.2f})")
        
        # Add the result to our *global* list
        self.triple.append(p_homogeneous)
            
        # Update title
        ax.set_title(f"Point '{point_name}' defined. Close this window to continue.")
        
        # We are done with this plot. Disconnect the event handler
        # and close the window.
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig) # This un-blocks the 'plt.show()' call in main()
        
        # Redraw the plot (briefly, before it closes)
        fig.canvas.draw()

    def get_image_lines(self):
        global fig, ax
        
        for line_name, image_path in zip(line_names, self.image_paths):
            try:
                image = plt.imread(image_path)
            except FileNotFoundError:
                print(f"Error: Image file not found at '{image_path}'")
                sys.exit(1)
            except Exception as e:
                print(f"Error loading image: {e}")
                sys.exit(1)
            image_name = os.path.basename(image_path)
            print(f"\nProcessing line '{line_name}' from image '{image_name}'...")
            # 3. Create the plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            ax.set_aspect('equal')
            ax.set_title(f"Click point 1 for line '{line_name}'")
            ax.set_xlabel("Click 2 points to define the line. Do not zoom or pan.")
            
            # 4. Create a state dictionary for this plot
            # This avoids complex globals and keeps each plot's state separate.
            plot_state = {
                "points": [],
                "line_name": line_name,
                "fig": fig,
                "ax": ax,
                "cid": None # Will store the connection ID
            }

            # 5. Connect the click event handler
            # We use a lambda function to pass our 'plot_state' to the handler
            cid = fig.canvas.mpl_connect(
                'button_press_event',
                lambda event: self.onclick_function(event, plot_state)
            )
            plot_state["cid"] = cid # Store the ID so we can disconnect it later

            # 6. Show the plot.
            # This call is BLOCKING. The script will pause here until
            # 'plt.close(fig)' is called (inside our onclick handler).
            plt.show() 
            
            print(f"Finished processing line '{line_name}'.")
    
    def get_image_names(self):
        if not os.path.isdir(self.data_directory):
            print(f"Error: Directory '{self.data_directory}' does not exist.")
            exit(1)
        image_paths = os.listdir(self.data_directory)
        image_paths = [os.path.join(self.data_directory, p) for p in image_paths if p.endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_paths) != 3:
            print("Error: Exactly 3 image paths must be in specified directory.")
            exit()
        return image_paths
        
    def line_line_line(self):
        self.get_image_lines()
        self.save_lines_to_file()

    def get_image_points(self):
        global fig, ax
        
        for point_name, image_path in zip(point_names, self.image_paths):
            try:
                image = plt.imread(image_path)
            except FileNotFoundError:
                print(f"Error: Image file not found at '{image_path}'")
                sys.exit(1)
            except Exception as e:
                print(f"Error loading image: {e}")
                sys.exit(1)
            image_name = os.path.basename(image_path)
            print(f"\nProcessing point '{point_name}' from image '{image_name}'...")
            # 3. Create the plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            ax.set_aspect('equal')
            ax.set_title(f"Click point 1 for point '{point_name}'")
            
            # 4. Create a state dictionary for this plot
            # This avoids complex globals and keeps each plot's state separate.
            plot_state = {
                "points": [],
                "point_name": point_name,
                "fig": fig,
                "ax": ax,
                "cid": None # Will store the connection ID
            }

            # 5. Connect the click event handler
            # We use a lambda function to pass our 'plot_state' to the handler
            cid = fig.canvas.mpl_connect(
                'button_press_event',
                lambda event: self.onclick_function(event, plot_state)
            )
            plot_state["cid"] = cid # Store the ID so we can disconnect it later

            # 6. Show the plot.
            # This call is BLOCKING. The script will pause here until
            # 'plt.close(fig)' is called (inside our onclick handler).
            plt.show() 

            print(f"Finished processing point '{point_name}'.")

    def point_point_point(self):
        self.get_image_points()
        self.save_points_to_file()

def main():
    if len(sys.argv) < 3:
        print("Usage: python correspondence_type directory_path")
        sys.exit(1)
    correspondence_type = sys.argv[1]
    directory_path = sys.argv[2]
    rank_computation = RankComputation(data_directory=directory_path, correspondence_type=correspondence_type)
    rank_computation.run()

if __name__ == "__main__":
    main()